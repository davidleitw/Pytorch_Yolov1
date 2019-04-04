import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def Convolutional3x3(Input, Output, stride=1):
    return nn.Conv2d(Input, Output, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    Expansion = 1

    def __init__(self, Input, Output, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = Convolutional3x3(Input, Output, stride=stride)
        self.bn1 = nn.BatchNorm2d(Output)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Convolutional3x3(Output, Output)
        self.bn2 = nn.BatchNorm2d(Output)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        temp = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)


        if self.downsample is not None:
            temp = self.downsample(x)

        output += temp
        output = self.relu(output)
        return output

class Bottleneck(nn.Module):
    Expansion = 4

    def __init__(self, Input, Output, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(Input, Output, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(Output)
        self.conv2 = nn.Conv2d(Output, Output, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(Output)
        self.conv3 = nn.Conv2d(Output, Output*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(Output*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        temp = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        if self.downsample is not None:
            temp = self.downsample(x)

        output = output+ temp
        output = self.relu(output)

        return output

class DarkNet(nn.Module):
    expansion = 1

    def __init__(self, Input, Output, stride=1, blocktype='A'):
        super(DarkNet, self).__init__()
        self.conv1 = nn.Conv2d(Input, Output, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(Output)

        self.conv2 = nn.Conv2d(Output, Output, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)
