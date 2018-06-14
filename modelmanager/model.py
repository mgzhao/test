from collections import OrderedDict
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as td
import torch.optim as optim
import torch.optim.lr_scheduler as lr
import torch.utils.data

from modelmanager.distributed import get_partition, init_processes
from modelmanager.loss import get_loss
from modelmanager.tflogger import Logger


def to_np(x):
        # Converts tensor to numpy
    return x.data.cpu().float().numpy()


class Model:
    metrics = {}

    def __init__(
            self, name, data, basepath=None,
            network_config={}, train_params={},
            resume=False,
            gpus=[0],
            loss='MSE',
            num_workers=0,
            half_precision=False,
            loss_scale=1,
            distributed=None):
        self.name = name
        self.num_workers = num_workers
        self.distributed = distributed
        self.use_gpu = len(gpus) > 0
        # half precision is not properly implemented for cpu models
        self.half_precision = half_precision and self.use_gpu
        self.loss_scale = loss_scale
        self._set_distributed_params(distributed)
        self._set_checkpoint_path(basepath)

        if resume:
            assert self.checkpoint_path
            self.load_data(**data)
            self.load_checkpoint(network_config)
            self.train_params.update(train_params)

        else:
            assert train_params
            assert network_config
            self._initialize_network(network_config, train_params, data)

        if self.half_precision:
            self.net = self.net.half()
            for layer in self.net.modules():
                self.convert_flagged_to_float(layer)
        if self.use_gpu:
            self.net = self.net.cuda()
        if self.distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(self.net)
        elif len(gpus) > 1:
            self.net = torch.nn.DataParallel(self.net)

        self.loss_fn = get_loss(loss)
        self._set_loggers(basepath)
        self._set_data_loaders()

    def convert_flagged_to_float(self, layer):
        if hasattr(layer, 'always_float'):
            layer.float()
        for child in layer.children():
            self.convert_flagged_to_float(child)
        return layer

    def _set_data_loaders(self):
        if self.distributed:
            traindata = get_partition(self.train_data)
        else:
            traindata = self.train_data
        print("Training set has {} samples.".format(len(traindata)))
        if self.validation_data:
            print("Validation set has {} samples.".format(
                len(self.validation_data)))
        else:
            print("No validation data")
        self.train_loader = torch.utils.data.DataLoader(
            traindata, batch_size=round(self.train_params['batch_size']),
            shuffle=True, pin_memory=True, num_workers=self.num_workers)
        if self.validation_data is None:
            self.validation_loader = None
        else:
            if self.distributed:
                evaldata = get_partition(self.validation_data)
            else:
                evaldata = self.validation_data
            self.validation_loader = torch.utils.data.DataLoader(
                evaldata,
                # batch_size=round(self.train_params['batch_size']),
                batch_size=2,
                drop_last=True,  # necessary to work around bug (issue# 5587)
                shuffle=False, pin_memory=True, num_workers=self.num_workers)

    def _initialize_network(self, network_config, train_params, data):
        self.initial_epoch = 0
        self.best_loss = math.inf
        self.train_params = {'optimizer_state': None}
        self.train_params.update(train_params)
        self.load_data(**data)
        self.init_network(**network_config)
        self.network_config = network_config
        self.initial_epoch = 0
        self.best_loss = math.inf

    def _set_loggers(self, basepath):
        self.loggers = {}
        if self.rank == 0 and basepath is not None:
            log_dir = Path(basepath) / self.name
            self.loggers['train'] = Logger(str(log_dir / 'train'))
            if self.validation_data:
                self.loggers['eval'] = Logger(str(log_dir / 'eval'))

    def _set_distributed_params(self, distributed):
        if distributed:
            self.distributed["group_name"] = self.name
            self.rank = init_processes(**self.distributed)
            self.world_size = distributed["world_size"]
        else:
            self.rank = 0

    def _set_checkpoint_path(self, basepath):
        if basepath is not None:
            prefix = Path(basepath)
            prefix = prefix / self.name
            prefix = prefix / "rank{}".format(self.rank)
            self.checkpoint_path = prefix
        else:
            self.checkpoint_path = None

    def setup_for_training(self):
        self.parameters = filter(
            lambda p: p.requires_grad, self.net.parameters())
        self.optimizer = optim.Adam(
            self.parameters, lr=self.train_params['learning_rate'])
        if self.train_params['optimizer_state']:
            self.optimizer.load_state_dict(
                self.train_params['optimizer_state'])

        if ('lr_schedule' in self.train_params
                and self.train_params['lr_schedule']):
            self.scheduler = lr.MultiStepLR(
                self.optimizer, milestones=self.train_params['lr_schedule'],
                gamma=self.train_params['lr_gamma'],
                last_epoch=self.initial_epoch-1)
        else:
            self.scheduler = lr.MultiStepLR(
                self.optimizer, milestones=[1e7],
                gamma=1.0
            )

    def __call__(self, x, training=False):
        if not training:
            self.net.eval()
        if self.half_precision:
            x = x.half()
        if self.use_gpu:
            x = x.cuda()
        return self.net(x)

    def train(self):
        self.setup_for_training()
        for epoch in range(self.initial_epoch, self.train_params['epochs']):
            tic = time.time()
            loss, rate = self.step(epoch)
            validation_loss, validation_metrics = self.measure_validation()
            if validation_loss:
                is_best = validation_loss < self.best_loss
                self.best_loss = min(validation_loss, self.best_loss)
            else:
                is_best = loss < self.best_loss
                self.best_loss = min(loss, self.best_loss)
            if self.checkpoint_path:
                self.save_checkpoint(epoch, is_best=is_best)
            info = {}
            info['train'] = {'rate': rate}
            info['eval'] = {}
            if validation_loss:
                info['eval']['loss'] = validation_loss
            for metric, measurement in validation_metrics.items():
                info['eval'][metric] = measurement
            self.log_epoch(epoch, info)
            how_long = time.time() - tic
            if self.rank == 0:
                msg = self._build_info_msg(
                    epoch, loss, rate, validation_loss, how_long)
                print(msg)

    def _build_info_msg(self, epoch, loss, rate, validation_loss, how_long):
        msg = 'Train Epoch: {:3}  Loss: {:2.6f}\t{:6.1f} samples/sec'
        msg = msg.format(epoch, loss, rate)
        if validation_loss is not None:
            msg += '\tEval loss: {:.6f}'.format(validation_loss)
        msg += '\tTime: {:2.1f} sec'.format(how_long)
        return msg

    def step(self, epoch):
        self.scheduler.step()
        self.net.train()
        steps_per_epoch = len(self.train_loader)
        batch_size = round(self.train_params['batch_size'])
        loss_history = []
        metric_history = {}
        for metric in self.metrics:
            metric_history[metric] = []
        tic = time.time()
        prev_logged_idx = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.half_precision:
                data = data.half()
            if self.use_gpu:
                data = data.cuda()
                target = target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.net(data)
            output = output.float()
            loss = self.loss_fn(output, target)
            loss *= self.loss_scale
            loss.backward()
            if self.loss_scale != 1:
                for param in self.parameters:
                    param.grad.data = param.grad.data/self.loss_scale
            self.optimizer.step()
            loss_history.append(loss.item())
            for metric, metric_fn in self.metrics.items():
                result = metric_fn(output, target)
                metric_history[metric].append(result.item())
            if batch_idx % 10 == 0 and self.loggers:
                step = (steps_per_epoch * epoch) + batch_idx
                step *= batch_size
                if self.distributed:
                    step *= self.world_size
                avg_loss = np.mean(loss_history[prev_logged_idx:batch_idx+1])
                info = {'train': {}}
                info['train']['loss'] = avg_loss
                for metric, metric_fn in self.metrics.items():
                    avg_metric = np.mean(
                        metric_history[metric][prev_logged_idx:batch_idx+1])
                    info['train'][metric] = avg_metric
                self.log_step(step, info)
                prev_logged_idx = batch_idx
        rate = len(self.train_data) / (time.time() - tic)
        return np.mean(loss_history), rate

    def measure_validation(self, dataloader=None):
        dataloader = dataloader or self.validation_loader
        if dataloader is None:
            return None, {}
        self.net.eval()
        cum_loss = 0
        cum_metrics = {}
        for metric in self.metrics:
            cum_metrics[metric] = 0
        for data, target in dataloader:
            if self.half_precision:
                data = data.half()
            if self.use_gpu:
                data = data.cuda()
                target = target.cuda()
            data, target = Variable(data), Variable(target)
            output = self.net(data)
            output = output.float()
            loss = self.loss_fn(output, target)
            cum_loss += loss.item()
            for metric, metric_fn in self.metrics.items():
                result = metric_fn(output, target)
                cum_metrics[metric] += result.item()
        cum_loss = cum_loss / max(len(dataloader), 1)
        for m in cum_metrics:
            cum_metrics[m] /= max(len(dataloader), 1)
        if self.distributed:
            x = torch.Tensor([cum_loss])
            td.all_reduce(x)
            cum_loss = x[0] / self.world_size
            for m in cum_metrics:
                x = torch.Tensor([cum_metrics[m]])
                td.all_reduce(x)
                cum_metrics[m] = x[0] / self.world_size
        return (cum_loss, cum_metrics)

    def log_step(self, step, info):
        if not self.loggers:
            return
        for logid, log in info.items():
            for tag, value in log.items():
                self.loggers[logid].scalar_summary(tag, value, step+1)

    def log_epoch(self, epoch, scalars):
        step = (epoch + 1) * len(self.train_data)
        if not self.loggers:
            return
        scalars['train']['epoch'] = epoch + 1
        self.log_step(step, scalars)
        # Log values and gradients of the parameters (histogram)
        for tag, value in self.net.named_parameters():
            tag = tag.replace('.', '/')
            if value.grad is not None:
                self.loggers['train'].histo_summary(tag, to_np(value), step+1)
                self.loggers['train'].histo_summary(
                    tag+'/grad', to_np(value.grad), step+1)

        # Log the visuals
        info = self.get_visuals()
        for logid, log in info.items():
            for tag, images in log.items():
                self.loggers[logid].image_summary(tag, images, step+1)

    def save_checkpoint(self, epoch, is_best=False, filename='latest.pt'):
        self.train_params['optimizer_state'] = self.optimizer.state_dict()
        state = {
            'epoch': epoch + 1,
            'name': self.name,
            'state_dict': self.net.state_dict(),
            'best_loss': self.best_loss,
            'network_config': self.network_config,
            'train_params': self.train_params,
        }
        filepath = self.checkpoint_path / filename
        parent = filepath.parent
        parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(filepath))
        if is_best:
            shutil.copy(
                str(filepath), str(self.checkpoint_path / 'best.pt'))

    def load_checkpoint(self, network_config={}, filename='latest.pt'):
        state = torch.load(str(self.checkpoint_path / filename))
        network_config.update(state['network_config'])
        self.init_network(**network_config)
        self.network_config = network_config
        self.best_loss = state['best_loss']
        self.initial_epoch = state['epoch']
        self.train_params = state['train_params']
        state_dict = OrderedDict()
        for k, v in state['state_dict'].items():
            if k[:7] == 'module.':
                name = k[7:]
            else:
                name = k
            state_dict[name] = v
        self.net.load_state_dict(state_dict)
        print("=> loaded checkpoint (epoch {})".format(state['epoch']))

    def init_network(self, network_config):
        raise NotImplementedError("Class needs to implement init_network")

    def load_data(self, **data):
        raise NotImplementedError("Class needs to implement load_data")

    def get_visuals(self):
        raise NotImplementedError("Class needs to implement get_visuals")

    def pbt_eval(self):
        raise NotImplementedError("Class needs to implement pbt_eval")
