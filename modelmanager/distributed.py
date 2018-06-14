import os
import random

import torch.distributed as td
from torch.multiprocessing import Process


def train(Model, model_args):
    # Run one worker node for each gpu
    gpus = model_args['gpus']
    model_args["distributed"]["world_size"] *= len(gpus)
    processes = []
    for gpu in gpus:
        p = Process(target=launch_worker_thread, args=(gpu, Model, model_args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def launch_worker_thread(gpu, Model, model_args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    model = Model(**model_args)
    model.train()


def init_processes(
    world_size, sharedfile,
    group_name, backend='gloo',
    initialized=False
):
    """ Initialize the distributed environment. """
    if not initialized:
        filepath = 'file://{}'.format(sharedfile)
        td.init_process_group(
            backend,
            init_method=filepath,
            group_name=group_name,
            world_size=world_size)
    rank = td.get_rank()
    return rank


class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        random.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def get_partition(dataset):
    size = td.get_world_size()
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(td.get_rank())
    return partition
