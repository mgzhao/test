import torch.nn as nn
import torch.nn.functional as F


class AbstractLayer():
    def __init__(self, *args, **kwargs):
        self.define_layers()
        super().__init__(*args, **kwargs)

    def define_layers(self):
        # Should define self.Conv, self.Norm, and self.Dropout
        raise NotImplementedError("define_layers is not defined")


class Layer2D(AbstractLayer):
    def define_layers(self):
        self.Conv = nn.Conv2d
        self.Dropout = nn.Dropout2d
        self.Norm = nn.BatchNorm2d


class Layer3D(AbstractLayer):
    def define_layers(self):
        self.Conv = nn.Conv3d
        self.Dropout = nn.Dropout3d
        self.Norm = nn.BatchNorm3d


class AbstractBasicBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        stride, nonlinearity=F.relu,
        dropout=0.0,
    ):
        super().__init__()
        self.nonlinearity = nonlinearity
        if dropout > 0.0:
            self.dropout = self.Dropout(p=dropout, inplace=True)
        else:
            self.dropout = lambda x: x

        self.alignment = lambda x: x
        self.conv1 = self.Conv(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        if stride > 1 or in_channels != out_channels:
            self.alignment = self.Conv(
                in_channels, out_channels,
                kernel_size=stride, stride=stride, bias=False
            )
        self.conv2 = self.Conv(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.norm1 = self.Norm(in_channels)
        self.norm2 = self.Norm(out_channels)

    def forward(self, x):
        original = x
        x = self.nonlinearity(self.norm1(x), inplace=True)
        x = self.conv1(x)
        x = self.nonlinearity(self.norm2(x), inplace=True)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + self.alignment(original)


class BasicBlock2D(Layer2D, AbstractBasicBlock):
    pass


class BasicBlock3D(Layer3D, AbstractBasicBlock):
    pass


class BlockStack(nn.Module):
    def __init__(self, in_channels, out_channels, Block=None,
                 stride=1, blocks=1, **kwargs):
        super().__init__()
        layers = [Block(in_channels, out_channels, stride, **kwargs)]
        for _ in range(1, blocks):
            layers.append(
                Block(out_channels, out_channels, stride=1, **kwargs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AbstractWideResNet(nn.Module):
    def __init__(self, in_channels, base_channels=16,
                 widening_factor=10, blocks=2,
                 strides=[2, 2, 2], **blockargs):
        super().__init__()
        c = base_channels
        k = widening_factor
        channels = [k * c * 2**n for n in range(len(strides))]
        self.out_channels = channels[-1]
        channels.append(c)
        entry_norm = self.Norm(in_channels)
        entry = self.Conv(
            in_channels, c, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        layers = [
            BlockStack(channels[n-1], channels[n],
                       stride=strides[n],
                       blocks=blocks, **blockargs)
            for n in range(len(strides))
        ]
        self.layers = nn.Sequential(entry_norm, entry, *layers)

    def forward(self, x):
        return self.layers(x)


class WideResNet2D(Layer2D, AbstractWideResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, Block=BasicBlock2D, **kwargs)


class WideResNet3D(Layer3D, AbstractWideResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, Block=BasicBlock3D, **kwargs)


class Classifier2D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.resnet = WideResNet2D(in_channels, **kwargs)
        self.norm = nn.BatchNorm2d(self.resnet.out_channels)
        self.class_weights = nn.Linear(self.resnet.out_channels, out_channels)

    def forward(self, x):
        x = self.norm(self.resnet(x))
        x = x.mean(2).mean(2)
        return self.class_weights(x)


class Benchmark3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.resnet = WideResNet3D(in_channels, widening_factor=8)
        c = self.resnet.out_channels
        self.up1 = nn.ConvTranspose3d(
            c, c//2, kernel_size=4,
            stride=2, padding=1
        )
        self.out_channels = c//2

    def forward(self, x):
        x = self.resnet(x)
        x = self.up1(x)
        return F.upsample(x, scale_factor=2, mode='trilinear')


class Classifier3D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.net = Benchmark3D(in_channels, **kwargs)
        self.norm = nn.BatchNorm3d(self.net.out_channels)
        self.class_weights = nn.Linear(self.net.out_channels, out_channels)

    def forward(self, x):
        x = self.norm(self.net(x))
        x = x.mean(2).mean(2).mean(2)
        return self.class_weights(x)
