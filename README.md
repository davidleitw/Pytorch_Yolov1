# Pytorch_Yolov1
> 這個專案主是要記錄用pytorch實作Yolo_v1的過程，會從資料預處理開始一直介紹到神經網路的搭建。

> 等這個專案結束會慢慢更新Yolo_v2, Yolo_v3並且做一個總結，大致介紹一下每一代做了什麼樣的改進。

> 2019/3/25 進度: 資料預處理的class的實作，其中也學到了很多影像的觀念會一一的介紹。

## 使用的資料集，以及資料預處理的部分:
*  -[x] 資料集介紹 Voc2012資料集
*  -[x] 讀入資料集 Xml讀入之後轉txt檔案敘述每張圖片的物件
*  -[x] 資料預處理 針對jpg資料集當中的影像搭配上面.txt檔案的描述來做檔案的預處理

### 資料集介紹 - Voc2012
`   下載網址: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html   `
        .
    └── VOCdevkit     # Root
    
        └── VOC2012   # 不同年份的資料集，這裡只下載了2012的，還有2007等其它年份的
        
            ├── Annotations        # 內容物為xml檔案，用途是描述資料集中每個Image的物件座標，物件編號等..
            
            ├── ImageSets          # 目錄底下存放的都是.txt檔案，檔案中每一行包含一個圖片的名稱，末尾會加上±1表示正負樣本
            
            │   ├── Action         # 存放的是人的動作，實作Yolo的時候比較不會用到這部分的內容
            
            │   ├── Layout         # 存放的每個影像中出現的人他的部分標示，實作Yolo也不會使用到
            
            │   ├── Main    
            
            │   └── Segmentation   # 存放的是可用於分割的資料,做檢測識別也是用不到的.
            
            ├── JPEGImages         # `最重要的部分，存放著我們要訓練的影像`
            
            ├── SegmentationClass  # 影像作分割之後的結果，如下圖所示
            
            └── SegmentationObject # 影像作分割之後的結果，如下圖所示
            
![SegmentationClass example picture](/Readme_ExampleImage/SegmentationClassExample.PNG)
