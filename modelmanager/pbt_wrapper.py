import torch
from torch.autograd import Variable


class PbtModel():
    def __init__(self, ModelClass, params):
        self.ModelClass = ModelClass
        params['gpus'] = [0]
        params['basepath'] = None
        self.params = params

    def build(self, logdir):
        self.model = self.ModelClass(**self.params)
        self.model._set_loggers(logdir)

    def step(self, n, t, train_params):
        self.model.train_params.update(train_params)
        self.model.setup_for_training()
        self.model.net.train()
        batches = n // round(self.model.train_params['batch_size'])
        assert batches <= len(self.model.train_loader)
        cum_loss = 0
        data_iter = iter(self.model.train_loader)
        for _ in range(batches):
            try:
                data, target = next(data_iter)
            except StopIteration:
                data_iter = iter(self.model.train_loader)
                data, target = next(data_iter)
            data = data.float().cuda()
            target = target.float().cuda()
            data, target = Variable(data), Variable(target)
            self.model.optimizer.zero_grad()
            output = self.model.net(data)
            loss = self.model.loss_fn(output, target)
            loss.backward()
            self.model.optimizer.step()
            cum_loss += loss.data[0]
        return cum_loss / batches

    def eval(self, t, train_params):
        return self.model.pbt_eval(t)

    def save(self, filename):
        state = {
            'name': self.model.name,
            'state_dict': self.model.net.state_dict(),
        }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.model.net.load_state_dict(state['state_dict'])
