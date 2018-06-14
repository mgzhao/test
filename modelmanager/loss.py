import torch
import torch.nn as nn

def get_loss(name):
    if name == 'MSE':
        return torch.nn.MSELoss()
    elif name == 'Relative':
        return  RelativeErrorLoss()
    elif name == 'L2':
        return  MeanL2Loss()
    elif name == 'RelativePlusMSE':
        return  RelativePlusMSELoss()
    else:
        raise NotImplementedError("Loss {} not recognized".format(name))

class L2Norm(nn.Module):
    def forward(self, x):
        x = x * x
        x = x.sum(1)
        return torch.sqrt(x)


class MeanL2Loss(nn.Module):
    def forward(self, input, target):
        L2 = L2Norm()
        norm = L2(input - target)
        return torch.mean(norm)


class RelativeErrorLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        super().__init__()

    def forward(self, input, target):
        L2 = L2Norm()
        target_norm = L2(target)
        error = L2(input - target)
        return torch.mean(error / (target_norm + self.epsilon))


class RelativePlusMSELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.relative = RelativeErrorLoss(epsilon)
        self.mse = torch.nn.MSELoss()

    def forward(self, input, target):
        return self.relative(input, target) + self.mse(input, target)
