import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

ModelsPath = {'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
              'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
              'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
              'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
              'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
              'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
              'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
              'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth', }

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, Features=None, NumClasses=20):
        super(VGG, self).__init__()
        self.features = Features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, NumClasses))
        self._InitWeight()

    def _InitWeight(self):
        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.sigmoid(x)
        x = x.view(-1, 7, 7, 30)
        return x

def MakeLayers(Cfg, Batch_Norm=None):
    Layerslist = []
    InputChannels = 3
    FirstFlag = True
    Stride = 1

    for char in Cfg:
        Stride = 1
        if char == 64 and FirstFlag == True:
            Stride = 2
            FirstFlag = False
        if char == 'M':
            # Layerslist.append(nn.MaxPool2d(kernel_size=2, stride=2))
            Layerslist += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            Conv = nn.Conv2d(InputChannels, char, kernel_size=2, stride=Stride, padding=1)
            if Batch_Norm == True:
                # Layerslist.append(Conv, nn.BatchNorm2d(char), nn.ReLU(inplace=True))
                Layerslist += [Conv, nn.BatchNorm2d(char), nn.ReLU(inplace=True)]
            else:
                # Layerslist.append(Conv, nn.ReLU(inplace=True))
                Layerslist += [Conv, nn.ReLU(inplace=True)]
            InputChannels = char

    return nn.Sequential(*Layerslist)

def Vgg19(Pretrained=False, **kwargs):
    Model = VGG(MakeLayers(cfg['E'], Batch_Norm=False), **kwargs)
    if Pretrained:
        Model.load_state_dict(model_zoo.load_url(ModelsPath['vgg19']))

def Vgg19Bn(Pretrained=False, **kwargs):
    Model = VGG(MakeLayers(cfg['E'], Batch_Norm=True), **kwargs)
    if Pretrained:
        Model.load_state_dict(model_zoo.load_url(ModelsPath['vgg19_bn']))
    return  Model



if __name__ == '__main__':
    Model = Vgg19(Pretrained=True)
    Model.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                     nn.ReLU(True),
                                     nn.Dropout(),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(True),
                                     nn.Dropout(),
                                     nn.Linear(4096, 7*7*30))
    print(Model)

    Img = torch.rand(2, 3, 224, 224)
    Output = Model(Img)
    print(Output.size())

    # print(Model)