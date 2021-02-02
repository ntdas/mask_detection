import numpy as np 
import cv2 as cv
import os
import xml.etree.ElementTree as ET


IMAGE_PATH = os.path.join('data','images')
ANN_PATH = os.path.join('data','annotations')
LABEL_PATH = os.path.join('.','data','label')
label_dict = {'with_mask':0, 'without_mask':1, 'mask_weared_incorrect':2}


def main():
    imlist = os.listdir(IMAGE_PATH)
    annlist = os.listdir(ANN_PATH)
    print('Number of images: ', len(imlist))
    debug = 0
    if not os.path.exists(LABEL_PATH):
        os.makedirs(LABEL_PATH)

    for name in annlist:
        I = cv.imread(os.path.join(IMAGE_PATH, name.replace('.xml','.png')))
        w = I.shape[1]
        h = I.shape[0]
        with open(os.path.join(LABEL_PATH,name.replace('.xml','.txt')),'w') as f:
            x = ET.parse(os.path.join(ANN_PATH,name))
            r = x.getroot()
            for child in r:
                if child.tag == 'filename':
                    print('Processing ',child.text)
                    
                if child.tag == 'object':
                    if debug == 1:  cur_label = []
                    for obj in child:
                        if obj.tag == 'name':
                            if debug == 1 :print('Label',obj.text)
                            objid = label_dict[obj.text]
                            if debug ==1 :cur_label.append(objid)
                            f.write(str(objid)+' ')
                        if obj.tag == 'bndbox':
                            for coor in obj:
                                if debug == 1: cur_label.append(coor.text)
                                temp = np.float(coor.text)
                                if coor.tag == 'xmin' or coor.tag =='xmax':
                                    temp = temp/w
                                else:
                                    temp = temp/h
                                f.write(str(temp)+' ')
                    if debug == 1: print(np.array(cur_label).astype('int'))
                    f.write('\n')
            
            if debug == 1 :print('========================')
    #with open(os.path.join('.',DATA_FOLDER,LABEL_PATH,name.replace('.xml','.txt'),'w') as f:


if __name__ == '__main__':
    main()