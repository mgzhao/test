import random
import torch

from modelmanager import Model, run

from resnet import Classifier2D, Classifier3D


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, dims=(3, 224, 224), classes=1000,  length=256):
        self.dims = dims
        self.length = length
        self.classes = classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.rand(self.dims), random.randrange(self.classes)


class BenchmarkModel(Model):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.loggers = {}

    def log_step(self, *args, **kwargs):
        pass

    def log_epoch(self, epoch, scalars):
        pass

    def get_visuals(self):
        return {}


class WideResNetModel(BenchmarkModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def load_data(self, dims=[3, 224, 224], **kwargs):
        self.dim = len(dims) - 1
        self.train_data = RandomDataset(dims=dims, **kwargs)
        self.validation_data = None

    def init_network(self, **network):
        input, _ = self.train_data[0]
        network['in_channels'] = input.shape[0]
        if self.dim == 2:
            self.net = Classifier2D(**network)
        elif self.dim == 3:
            self.net = Classifier3D(**network)
        else:
            raise ValueError("Only 2 or 3 dimensional input allowed")


models = {
    "WideResNet": WideResNetModel,
}

if __name__ == "__main__":
    run(models)
