import torch
import numpy as np
import os

class YoloConfig():
    ProgName  = 'Yolo v1'
    TrainImageRoot = r'D:\MyYoLo\VOCdevkit\VOC2012\JPEGImages/   ' # Img
    TrainAnnotations = r'D:\MyYoLo\VOCdevkit\VOC2012\Annotations/' # Xml
    TrainData = r'D:\MyYoLo\Voc2012.txt' # Come from Xml file
    Testfile = ''

    A = 15
    Using_GPU = True if torch.cuda.is_available() else False
    BatchSize = 32
    EpochsNum = 100
    LearningRate = 0.001

    Voc_Classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


Cfg = YoloConfig()


