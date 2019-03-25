# Pytorch_Yolo_V1
## 使用環境
1. Python 3.6.6
2. OpenCv 
3. Pytorch 0.4.0
***
> 這個專案主是要記錄用pytorch實作Yolo_v1的過程，會從資料預處理開始一直介紹到神經網路的搭建。

> 等這個專案結束會慢慢更新Yolo_v2, Yolo_v3並且做一個總結，大致介紹一下每一代做了什麼樣的改進。

> 2019/3/25 進度: 資料預處理的class的實作，其中也學到了很多影像的觀念會一一的介紹。

***

## 使用的資料集，以及資料預處理的部分:
*  1. -[x] 資料集介紹-> Voc2012資料集
*  2. -[x] 讀入資料集-> Xml讀入之後轉txt檔案敘述每張圖片的物件
*  3. -[x] 資料預處理-> 針對jpg資料集當中的影像搭配上面.txt檔案的描述來做檔案的預處理

### 1.資料集介紹 - Voc2012

這個資料集出自於PASCAL VOC挑戰賽（The PASCAL Visual Object Classes)是一個世界級的計算機視覺挑戰賽。 

PASCAL全稱：Pattern Analysis, Statical Modeling and Computational Learning，是一個由歐盟資助的網絡組織。

此資料集在分類，定位，檢測，分割，動作識別領域都有人做出很知名的模型(Rcnn, Yolo...)

PASCAL VOC從2005年開始舉辦挑戰賽，每年的內容都有所不同，從最開始的分類，到後面逐漸增加檢測，分割，人體佈局，動作識別（Object Classification 、Object Detection、Object Segmentation、Human Layout、 Action Classification）等等關於不同應用所需要的資料。



`   下載網址: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html   `
        
        └── VOCdevkit # Root
    
        └── VOC2012   # 不同年份的資料集，這裡只下載了2012的，還有2007等其它年份的
        
            ├── Annotations        # 內容物為xml檔案，用途是描述資料集中每個Image的物件座標，物件編號等..
            
            ├── ImageSets          # 目錄底下存放的都是.txt檔案，檔案中每一行包含一個圖片的名稱，末尾會加上±1表示正負樣本
            
            │   ├── Action         # 存放的是人的動作，實作Yolo的時候比較不會用到這部分的內容
            
            │   ├── Layout         # 存放的每個影像中出現的人他的部分標示，實作Yolo也不會使用到
            
            │   ├── Main    
            
            │   └── Segmentation   # 存放的是可用於分割的資料,做檢測識別也是用不到的.
            
            ├── JPEGImages         # 最重要的部分，存放著我們要訓練的影像
            
            ├── SegmentationClass  # 影像作分割之後的結果，如下圖所示
            
            └── SegmentationObject # 影像作分割之後的結果，如下圖所示
            
![SegmentationClass example picture](/Readme_ExampleImage/SegmentationClassExample.PNG)

下面是Annotations隨便取出的一個.xml檔案，以下會用註解的方式說明他如何敘述一張影像。

        <annotation>
                <folder>VOC2012</folder>                            # 說明此張對應的影像出自哪一年的資料集
                <filename>2007_000027.jpg</filename>                # 此張影像的名稱
                <source>                                            # source代表圖片來源
                        <database>The VOC2007 Database</database> 
                        <annotation>PASCAL VOC2007</annotation>
                        <image>flickr</image>
                </source>                                           # source區域結束
                <size>                                              # 影像大小
                        <width>486</width>
                        <height>500</height>
                        <depth>3</depth>
                </size>
                <segmented>0</segmented>                            # 是否分割
                <object>                                            # 物件說明區域
                        <name>person</name>                         # 物件名稱
                        <pose>Unspecified</pose>                    # 拍攝角度
                        <truncated>0</truncated>                    # 目標是否被截斷(物件是否有超出影像), 或者被遮擋（超過15%）
                        <difficult>0</difficult>                    # 檢測難易程度，這個主要是根據目標的大小，光照變化，圖片質量來判斷
                        <bndbox>                                    # 物件左上角跟右下角的點座標(xmin, ymin, xmax, ymax)
                                <xmin>174</xmin>
                                <ymin>101</ymin>
                                <xmax>349</xmax>
                                <ymax>351</ymax>
                        </bndbox>
                        <part>
                                <name>head</name>
                                <bndbox>
                                        <xmin>169</xmin>
                                        <ymin>104</ymin>
                                        <xmax>209</xmax>
                                        <ymax>146</ymax>
                                </bndbox>
                        </part>
                        <part>
                                <name>hand</name>
                                <bndbox>
                                        <xmin>278</xmin>
                                        <ymin>210</ymin>
                                        <xmax>297</xmax>
                                        <ymax>233</ymax>
                                </bndbox>
                        </part>
                        <part>
                                <name>foot</name>
                                <bndbox>
                                        <xmin>273</xmin>
                                        <ymin>333</ymin>
                                        <xmax>297</xmax>
                                        <ymax>354</ymax>
                                </bndbox>
                        </part>
                        <part>
                                <name>foot</name>
                                <bndbox>
                                        <xmin>319</xmin>
                                        <ymin>307</ymin>
                                        <xmax>340</xmax>
                                        <ymax>326</ymax>
                                </bndbox>
                        </part>
                </object>
        </annotation>

> 更詳細的資料請參考
> * [Pascal Voc資料集詳細介紹](https://arleyzhang.github.io/articles/1dc20586/)

***
### 2.讀入xml檔案轉換成.txt檔方便做預處理  
File : XML_to_Txt.py

File_path: Pytorch_Yolov1/MyYoLo/Yolo_v1/XML_to_Txt.py

```python
import os
from Config import Cfg
from xml.etree import ElementTree


def XmlToTxt(filename):
    Tree = ElementTree.parse(filename)
    Object = []
    for Obj in Tree.findall('object'):
        ObjStruct = {}
        ObjStruct['name'] = Obj.find('name').text
        Boxes = Obj.find('bndbox')
        ObjStruct['Boxes'] = [int(float(Boxes.find('xmin').text)),
                              int(float(Boxes.find('ymin').text)),
                              int(float(Boxes.find('xmax').text)),
                              int(float(Boxes.find('ymax').text))]
        Object.append(ObjStruct)
    return Object
```

上方程式碼是XML_to_Txt.py 中最關鍵的函式，XmlToTxt主要的用途就是把上一單元介紹的.xml檔轉成較好處理的.txt檔案。


```python
Tree = ElementTree.parse(filename)
```

一開始建立一個專門用來處理Xml檔案的tree，去做到搜索以及讀取我們所需的資料，之後利用.find()先找到'<bndbox></bndbox>'這種在.xml檔案出現的格式，並且使用.text將讀進來的東西轉成字串，藉此處理xml裡面的每一個物件標籤，讓他變成有規則的.txt檔案。

```python
for Obj in Tree.findall('object'):
        ObjStruct = {}
        ObjStruct['name'] = Obj.find('name').text
        Boxes = Obj.find('bndbox')
        ObjStruct['Boxes'] = [int(float(Boxes.find('xmin').text)),
                              int(float(Boxes.find('ymin').text)),
                              int(float(Boxes.find('xmax').text)),
                              int(float(Boxes.find('ymax').text))]
        Object.append(ObjStruct)
```

此段程式碼是找到所有帶有'object'的標籤，並且建立一個字典，使得每一個物件都有對應的名稱以及座標。

> 關於python對於xml的處理，若是想了解更多請參考下列網址
> * [Python xml.etree.ElementTree — The ElementTree XML API](https://docs.python.org/2/library/xml.etree.elementtree.html)




