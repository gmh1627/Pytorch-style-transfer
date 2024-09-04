# transform_pytorch.py

import torch
import torch.nn as nn

class TransformNet(nn.Module):
    def __init__(self):
        super(TransformNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.resid3 = ResidualBlock(128)
        self.resid4 = ResidualBlock(128)
        self.resid5 = ResidualBlock(128)
        self.conv_t1 = ConvTransposeLayer(128, 64, 3, 2)
        self.conv_t2 = ConvTransposeLayer(64, 32, 3, 2)
        self.conv_t3 = ConvLayer(32, 3, 9, 1, relu=False)

    def forward(self, x):
        y = x / 255.0
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.resid1(y)
        y = self.resid2(y)
        y = self.resid3(y)
        y = self.resid4(y)
        y = self.resid5(y)
        y = self.conv_t1(y)
        y = self.conv_t2(y)
        y = self.conv_t3(y)
        y = torch.tanh(y)
        y = x + y
        return torch.tanh(y) * 127.5 + 255. / 2

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = relu

    def forward(self, x):
        x = self.conv2d(x)
        x = self.instance_norm(x)
        if self.relu:
            x = torch.relu(x)
        return x

class ConvTransposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTransposeLayer, self).__init__()
        padding = kernel_size // 2
        self.conv2d_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=stride-1)
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv2d_transpose(x)
        x = self.instance_norm(x)
        return torch.relu(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.conv2 = ConvLayer(channels, channels, 3, 1, relu=False)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual