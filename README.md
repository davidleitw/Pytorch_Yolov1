# Pytorch_Yolo_V1
## 使用環境
1. Python 3.6.6
2. OpenCv 
3. Pytorch 0.4.0
***
> 這個專案主是要記錄用pytorch實作Yolo_v1的過程，會從資料預處理開始一直介紹到神經網路的搭建。

> 等這個專案結束會慢慢更新Yolo_v2, Yolo_v3並且做一個總結，大致介紹一下每一代做了什麼樣的改進。

> 2019/3/25 進度: 資料預處理的class的實作，其中也學到了很多影像的觀念會一一的介紹。

## 使用的資料集，以及資料預處理的部分:
*  -[x] 資料集介紹-> Voc2012資料集
*  -[x] 讀入資料集-> Xml讀入之後轉txt檔案敘述每張圖片的物件
*  -[x] 資料預處理-> 針對jpg資料集當中的影像搭配上面.txt檔案的描述來做檔案的預處理

***
### 資料集介紹 - Voc2012

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





