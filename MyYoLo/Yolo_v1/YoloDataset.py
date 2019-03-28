import os
import Config
import sys
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data


class Yolo_Dataset(data.Dataset):
    ImgSize = 224

    def __init__(self, listfile, root, train=None, transform=None):
        # print('Initialization DataSet')
        self.fnames = []
        self.Boxes = []
        self.labels = []
        self.mean = (123, 117, 104)
        self.Datas = []
        self.root = root
        self.train = train
        self.transform = transform

        with open(listfile) as f:
            lines = f.readlines()

        print('File lines number = {}'.format(len(lines)))

        # Reading Voc2012.txt
        for step, line in enumerate(lines):
            Data = line.strip().split()
            self.Datas.append(Data)  # 原始資料
            self.fnames.append(Data[0])  # .jpg檔名
            ObjectNumber = int(Data[1])  # 物件數量
            boxes = []
            label = []

            for i in range(ObjectNumber):
                temp = i + 1
                Class = float(Data[temp * 5 - 3])
                x1 = float(Data[temp * 5 - 2])
                y1 = float(Data[temp * 5 - 1])
                x2 = float(Data[temp * 5])
                y2 = float(Data[temp * 5 + 1])

                boxes.append([x1, y1, x2, y2])
                label.append(int(Class))

            self.Boxes.append(torch.Tensor(boxes))
            self.labels.append(torch.Tensor(label))
        self.Num_Img = len(self.Boxes)

    def __getitem__(self, Index):
        frame = self.fnames[Index]
        Img = cv2.imread(os.path.join(self.root + frame))
        Boxes = self.Boxes[Index].clone()
        Label = self.labels[Index].clone()

        if self.train:
            Img, Boxes = self.Random_Flip(Img, Boxes) # 隨機翻轉
            Img, Boxes = self.Random_Scale(Img, Boxes)# 固定高度 隨機伸縮寬度
            Img = self.Random_Blur(Img)
            Img = self.Random_Brightness(Img)
            Img = self.Random_Hue(Img)
            Img = self.Random_Saturation(Img)
            Img, Boxes, Label = self.Random_Shift(Img, Boxes, Label)

        Height, Width, _ = Img.shape
        Boxes = Boxes / torch.Tensor([Width, Height, Width, Height]).expand_as(Boxes)
        Img = self.BGR2RGB(Img)
        Img = self.SubMean(Img, self.mean)
        Img = cv2.resize(Img, (self.ImgSize, self.ImgSize))
        # print('Img shape = {}\nImg matrix = {}'.format(Img.shape, Img))
        #Target = self.Encoder(Boxes, Label)

        #for t in self.transform:
            #pass

        #return Img, Target


    def Encoder(self, Boxes, Labels):
        Target = torch.zeros((7, 7, 30))
        print(Target)
        CeilSize = float(1/7)
        Wh = Boxes[:, 2:] - Boxes[:, :2]
        Center_xy = (Boxes[:, 2:] + Boxes[:, :2])/2

        for Index in range(Center_xy.shape[0]):
            SampleCenter = Center_xy[Index]
            print(SampleCenter)
            ij = (SampleCenter/CeilSize).ceil()-1
            Target[int(ij[1]), int(ij[0]), 4] = 1
            Target[int(ij[1]), int(ij[0]), 4] = 1
            Target[int(ij[1]), int(ij[0]), int(Labels[Index]+9)] = 1
            

        return Target

    # 隨機調整亮度
    def Random_Brightness(self, BGR):
        if random.random() < 0.6:
            HSV = self.BGR2HSV(BGR)
            h, s, v = cv2.split(HSV)
            Random = random.choice([0.5, 1.5])
            v = np.clip(v * Random, 0, 255).astype(HSV.dtype)
            Hsv = cv2.merge((h, s, v))
            BGR = self.HSV2BGR(Hsv)
        return BGR

    # 隨機調整飽和度
    def Random_Saturation(self, BGR):
        if random.random() < 0.6:
            HSV = self.BGR2HSV(BGR)
            h, s, v = cv2.split(HSV)
            Random = random.choice([0.5, 1.5])
            s = np.clip(s * Random, 0, 255).astype(HSV.detype)
            HSV = cv2.merge((h, s, v))
            BGR = self.HSV2BGR(HSV)
            return BGR

    # 隨機調整色相
    def Random_Hue(self, BGR):
        if random.random() < 0.6:
            HSV = self.HSV2BGR(BGR)
            h, s, v = cv2.split(HSV)
            Random = np.random.choice([0.5, 1.5])
            h = np.clip(h * Random, 0, 255).astype(HSV.detype)
            HSV = cv2.merge((h, s, v))
            BGR = self.HSV2BGR(HSV)
            return BGR

    # 隨機模糊
    def Random_Blur(self, BGR):
        if random.random() < 0.6:
            BGR = cv2.blur(BGR, (5, 5))
        return BGR

    def Random_Shift(self, BGR, Boxes, Labels):
        Center = (Boxes[:, 2:] + Boxes[:, :2]) / 2
        if random.random() < 0.6:
            Height, Width, C = BGR.shape
            After_ShiftImg = np.zeros((Height, Width, C), dtype=BGR.dtype)
            After_ShiftImg[:, :, :] = (104, 117, 123)
            # print(After_ShiftImg)
            Shift_x = np.random.uniform(-Width * 0.2, Width * 0.2)
            Shift_y = np.random.uniform(-Height * 0.2, Height * 0.2)

            if Shift_x >= 0 and Shift_y >= 0:
                After_ShiftImg[int(Shift_y):, int(Shift_x):, :] = BGR[:Height - int(Shift_y), :Width - int(Shift_x), :]
            elif Shift_x >= 0 and Shift_y < 0:
                After_ShiftImg[:Height + int(Shift_y), int(Shift_x):, :] = BGR[-int(Shift_y):, :Width - int(Shift_x), :]
            elif Shift_x < 0 and Shift_y >= 0:
                After_ShiftImg[int(Shift_y):, :Width + int(Shift_x), :] = BGR[:Height - int(Shift_y), -int(Shift_x):, :]
            elif Shift_x < 0 and Shift_y < 0:
                After_ShiftImg[:Height + int(Shift_y), :Width + int(Shift_x), :] = BGR[-int(Shift_y):, -int(Shift_x):,
                                                                                   :]

            Shiftxy = torch.Tensor([int(Shift_x), int(Shift_y)]).expand_as(Center)
            Center = Center + Shiftxy
            Mark1 = (Center[:, 0] > 0) & (Center[:, 0] < Width)
            Mark2 = (Center[:, 1] > 0) & (Center[:, 1] < Height)
            Mark = (Mark1 & Mark2).view(-1, 1)
            Box_In = Boxes[Mark.expand_as(Boxes)].view(-1, 4)
            print('Box_In.shape = {}'.format(Box_In.shape))
            if len(Box_In) == 0:
                return BGR, Boxes, Labels

            Box_Shift = torch.Tensor([int(Shift_x), int(Shift_y), int(Shift_x), int(Shift_y)]).expand_as(Box_In)
            Box_In = Box_In + Box_Shift
            Label_In = Labels[Mark.view(-1)]
            '''
            print('Shiftx = {}, Shifty = {}'.format(Shift_x, Shift_y))
            print(type(After_ShiftImg), After_ShiftImg.shape)
            print('type of Shiftxy = {}'.format(type(Shiftxy)))
            print('Shiftxy = {}'.format(Shiftxy))
            print('type of Center = {}'.format(type(Center)))
            print('shape of Center = {}'.format(Center.shape))
            print('Center = ', Center)
            print('(Center[:, 0] > 0) = {}'.format((Center[:, 0] > 0)))
            print('(Center[:, 0] < Width) = {}'.format((Center[:, 0] < Width)))
            print('(Center[:, 1] > 0) = {}'.format((Center[:, 1] > 0)))
            print('(Center[:, 1] < Height) = {}'.format((Center[:, 1] < Height)))
            print('Mark1 = {} Mark2 = {}'.format(Mark1, Mark2))
            print('(Mark1 & Mark2) = {}'.format((Mark1 & Mark2)))
            print('Mark = {}'.format(Mark))
            '''
            return After_ShiftImg, Box_In, Label_In

        return BGR, Boxes, Labels

    def Random_Crop(self, BGR, Boxes, Lables):
        if random.random() < 0.6:
            pass

    def Random_Flip(self, Img, Boxes):

        if random.random() < 0.6:
            Imglr = np.fliplr(Img).copy()
            H, W, _ = Img.shape
            xmin = W - Boxes[:, 2]
            xmax = W - Boxes[:, 0]
            Boxes[:, 0] = xmin
            Boxes[:, 2] = xmax
            return Imglr, Boxes
        return Img, Boxes

    def Random_Bright(self, Img, delta=16):
        temp = random.random()
        if temp > 0.3:
            Img = Img * 0.3 + random.randrange(-delta, delta)
            Img = Img.clip(min=0, max=255).astype(np.uint8)
        return Img

    def Random_Scale(self, BGR, Boxes):
        if random.random() < 0.6:
            Scale = random.uniform(0.5, 1.5)
            Height, Width, C = BGR.shape
            BGR = cv2.resize(BGR, (int(Width * Scale), Height))
            ScaleTensor = torch.Tensor([Scale, 1, Scale, 1]).expand_as(Boxes)
            Boxes = Boxes * ScaleTensor
            return BGR, Boxes
        return BGR, Boxes

    def SubMean(self, BGR, Mean):
        _Mean = np.array(Mean, dtype=np.float32)
        BGR = BGR - _Mean
        return BGR

    def __GetTotalImgNumber(self):
        return self.Num_Img

    def BGR2RGB(self, Img):
        return cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, Img):
        return cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, Img):
        return cv2.cvtColor(Img, cv2.COLOR_HSV2BGR)

if __name__ == '__main__':

    Cfg = Config.YoloConfig()

    DataSet = Yolo_Dataset(listfile=Cfg.TrainData, root=Cfg.TrainImageRoot)
    Img = cv2.imread(r'D:\Pytorch_Yolov1\MyYoLo\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg')
    cv2.imshow('Img', Img)
    cv2.waitKey(0)
    print(DataSet.Boxes[1], DataSet.labels[1])
    try:
        print('Input img name = {}, shape = {}'.format(DataSet.fnames[1], Img.shape))
        print('Image object boxes = {}, shape = {}'.format(DataSet.Boxes[1], DataSet.Boxes[1].shape))

        Maxpoint = DataSet.Boxes[1][:, 2:]
        Minpoint = DataSet.Boxes[1][:, :2]
        print('Max = {}\nMin = {}'.format(Maxpoint, Minpoint))
        Cxy = (Maxpoint + Minpoint)/2
        print('Center = {}'.format(Cxy))
        print('CenterSize = {}'.format(Cxy.shape))
        print('{}\n{}'.format(Cxy.size()[0], type(Cxy)))
        DataSet.__getitem__(1)
        DataSet.Encoder(DataSet.Boxes[1], DataSet.labels[1])



        # Img2, b, l = DataSet.Random_Shift(Img, DataSet.Boxes[1], DataSet.labels[1])
        # Img2, _ = DataSet.Random_Flip(Img, DataSet.Boxes[0])
        # Img2 = DataSet.Random_Bright(Img)
        # Img2, _ = DataSet.Random_Scale(Img, DataSet.Boxes[1])
        # cv2.imshow('Img2', Img2)
        # print(Img2.shape)
        # cv2.waitKey(0)
    except BaseException:
        print('Error: Do not read the img\n')
