import torch
import torch.nn as nn
import math
import torch.nn.functional as F

ModelsPath = {'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
              'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
              'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
              'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
              'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
              'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
              'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
              'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',}

class VGG(nn.Module):
    def __init__(self, Features, NumClasses=20):
        super(VGG, self).__init__()
        self.features = Features
        self.classifier = nn.Sequential(nn.Linear(512*7*7, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, NumClasses))

    def _InitWeight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                pass



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.sigmoid(x)
        x = x.view(-1, 7, 7, 30)
        return x