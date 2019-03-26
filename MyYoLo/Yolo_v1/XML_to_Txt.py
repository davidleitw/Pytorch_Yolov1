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


if __name__ == '__main__':

    Writefile = open(r'D:\MyYoLo\Voc2012.txt', 'w')
    Annotations = Cfg.TrainAnnotations
    os.chdir(Annotations)
    Xmlfiles = os.listdir(Annotations)

    for Num, Xmlfile in enumerate(Xmlfiles):
        Img = Xmlfile.split('.')[0] + '.jpg '
        Path = Annotations + Xmlfile
        Results = XmlToTxt(Path)
        Num_Obj = len(Results)

        Writefile.write(Img)
        Writefile.write(str(Num_Obj) + ' ')

        for Result in Results:
            Boxes = Result['Boxes']
            ClassesName = Cfg.Voc_Classes.index(Result['name'])
            Writefile.write('{} {} {} {} {} '.format(str(ClassesName), str(Boxes[0]),
                                                     str(Boxes[1]), str(Boxes[2]),
                                                     str(Boxes[3])))

        Writefile.write('\n')

    print('Finish writeing, Img number = {}'.format(len(Xmlfiles)))
    print('Write into Voc2012.txt, Next step is train our yolo')
    Writefile.close()

'''

    每一行的格式:
    影像名稱 + 影像內所含的物體個數(Num) + (物體所代表的index + 4個座標點)
    所以每行共有 1 + 1 + 5*Num = (2+5*N)個數字

'''
