import os

import tqdm
from PIL import Image
import numpy as np
from scipy.io import loadmat, savemat
import xml.etree.ElementTree as ET
from utils.utils import get_classes


def get_data_mat(path, list_file, img_out_path):
    name = path.split('/')[-1]
    img_out_path = img_out_path + '/' + name
    os.makedirs(img_out_path, exist_ok=True)
    path_img1 = path + '/' + name + 'img'
    path_img2 = path + '/' + name + 'imgr'
    path_label = path + '/' + name + 'labelr'
    label_names = []
    for file in tqdm.tqdm(os.listdir(path_img1)):
        label_file = file.replace('.jpg', '.xml')
        if save_mat:
            img1 = np.array(Image.open(path_img1 + '/' + file).convert('RGB'))
            img2 = np.array(Image.open(path_img2 + '/' + file).convert('RGB'))
            assert img1.shape == img2.shape
            savemat(img_out_path + '/' + file.replace('.jpg', '.mat'), {'img1': img1, 'img2': img2})
        list_file.write(img_out_path + '/' + file.replace('.jpg', '.mat'))
        tree = ET.parse(path_label + '/' + label_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            difficult = 0
            if obj.find('difficult') != None:
                difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls == 'feright car':
                cls = 'feright_car'
            if cls not in label_names:
                label_names.append(cls)
            if cls not in classes or int(difficult) == 1:
                print(cls)
                continue
            cls_id = classes.index(cls)
            try:
                xmlbox = obj.find('polygon')
                b = (int(float(xmlbox.find('x1').text)), int(float(xmlbox.find('y1').text)),
                     int(float(xmlbox.find('x2').text)), int(float(xmlbox.find('y2').text)),
                     int(float(xmlbox.find('x3').text)), int(float(xmlbox.find('y3').text)),
                     int(float(xmlbox.find('x4').text)), int(float(xmlbox.find('y4').text)),
                     )
            except:
                try:
                    xmlbox = obj.find('bndbox')
                    b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
                         int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymax').text)),
                         int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymin').text)),
                         int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)),
                         )
                except:
                    continue
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        list_file.write('\n')
    print(label_names)


if __name__ == '__main__':
    classes_path = './model_data/ssdd_classes.txt'
    classes, _ = get_classes(classes_path)
    path_in = '/media/zyj/data/zyj/11/多模态融合检测'
    path_train = path_in + '/train'
    path_val = path_in + '/val'
    path_test = path_in + '/test'

    path_out = '/media/zyj/data_all/zyj/rgb融合检测/data_use'
    os.makedirs(path_out, exist_ok=True)

    save_mat = True

    ftrain = open('./train.txt', 'w')
    fval = open('./val.txt', 'w')
    ftest = open('./test.txt', 'w')

    get_data_mat(path_train, ftrain, path_out)
    get_data_mat(path_val, fval, path_out)
    get_data_mat(path_test, ftest, path_out)