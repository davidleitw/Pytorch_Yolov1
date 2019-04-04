import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

def GetTestImage(Filepath=None):
    Img = cv2.imread(Filepath)
    Img = cv2.resize(Img, (416, 416))
    Img = Img[:, :, ::-1].transpose((2, 0, 1))
    Img = Img[np.newaxis, :, :, :] / 255.0
    Img = torch.Tensor(Img).float()
    return Img

def LoadCfg(CfgFile=None):
    Cfg = open(CfgFile, 'r')
    lines = Cfg.read().split('\n')
    lines = [x for x in lines if len(x) > 0 and x[0] != '#']

    Block = {}
    Blocks = []
    for line in lines:
        if line[0] == '[':
            if len(Block) != 0:
                Blocks.append(Block)
                Block.clear()
            Block['type'] = line[1:-1].rstrip()
        else:
            Key, Value = line.split('=')
            Block[Key.rstrip()] = Value.lstrip()
    Blocks.append(Block)

    print(Blocks)
    return Blocks




if __name__ == '__main__':
    path = r'D:\YOLO_v3_tutorial_from_scratch\cfg\yolov3.cfg'
    LoadCfg(path)


