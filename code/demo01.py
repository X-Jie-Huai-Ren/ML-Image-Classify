"""
@autuor xuan
@email 1920425406@qq.com

图像处理
"""

from concurrent.futures import process
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, data, io
import os
import json

OUTPUT_TRAIN_IMG = 'D:\Programme WorkSpace\Python-Space\ML-design\processed_data//train_img.npy'
OUTPUT_TRAIN_LABEL = 'D:\Programme WorkSpace\Python-Space\ML-design\processed_data//train_label.npy'
OUTPUT_TRAIN_LABEL2 = 'D:\Programme WorkSpace\Python-Space\ML-design\processed_data//train_label2.npy'

OUTPUT_TEST_IMG = 'D:\Programme WorkSpace\Python-Space\ML-design\processed_data//test_img.npy'
OUTPUT_TEST_LABEL = 'D:\Programme WorkSpace\Python-Space\ML-design\processed_data//test_label.npy'





# 获取存放不同种类图片文件夹的路径
def get_folder_path(path):

    """
    功能: 获取各文件夹的路径
    """

    # 文件夹路径列表
    folder_path_lst = []

    try:
        folders = os.listdir(path)
    except Exception as error:
        print(error)

    # 遍历
    for folder in folders:

        folder_path_lst.append(os.path.join(path, folder))

    return folder_path_lst




# 获取图片路径和标签
def load_samples(folder_path, i):

    """
    功能: 得到图片路径以及标签
    """

    # 图片完整路径列表
    image_path_lst = []


    # 对应标签列表
    image_labels_lst = []

    try:
        files = os.listdir(folder_path)
    except Exception as error:
        print(error)


    # 遍历文件夹
    for file in files:

        # 如果是图片文件， 添加到图片路径列表
        if file.endswith('.jpeg'):
           image_path_lst.append(os.path.join(folder_path, file))

        # 若是json文件，读取json文件中图片的标签，并添加到标签列表
        else:
            file_json_path = os.path.join(folder_path, file)
            f = open(file_json_path, encoding='utf-8')


            json_data = json.load(f)
            # image_labels_lst.append(json_data['ProductCode'])
            image_labels_lst.append(i)


    return image_path_lst, image_labels_lst


# 将图片转化为RGB通道三维矩阵
def decode_image(image_path_lst, image_labels_lst): 

    """
    功能: 读取图片路径，并转为RGB通道三维矩阵
    """

    # 存放转化后的三维矩阵
    image_data_lst = []

    for image_path in image_path_lst:

        # 读取该路径下的图片
        # image = tf.io.read_file(image_path) 
        image = io.imread(image_path)

        # 图像缩放，返回矩阵
        dst = transform.resize(image, (70,90))

        # 将图片进行编码
        # x = tf.image.decode_jpeg(dst, channels=3)

        # 将tensor对象转化为np.ndarray类型
        # x_array = np.array(x)

        image_data_lst.append(dst)

        # break
    
    # 将列表转为np.ndarray对象
    img = np.array(image_data_lst)
    label = np.array(image_labels_lst)

    return img, label


# 最终数据存储
def data_store(folder_path_lst):

    """
    将处理后的图像数据存储在文件中
    """

    # 定义一个空的四维矩阵，用来存放图像数据。维度为: 样本数， 高， 宽， 通道数
    processed_img = np.empty([0, 70, 90, 3])

    # 定义一个空的一维矩阵， 存放对应的图片标签
    processed_label = []

    i = 0
    for folder_path in folder_path_lst:

        # 获取该文件夹下的图片标签
        image_paths_lst, image_labels_lst =  load_samples(folder_path, i)

        # 将图像编码为RGB三维矩阵形式
        image_rgb, label= decode_image(image_paths_lst, image_labels_lst)

        # 合并
        processed_img = np.vstack((processed_img, image_rgb))
        processed_label.extend(list(label))

        i = i+1


    
    processed_label = np.array(processed_label)

    return processed_img, processed_label
        
            



if __name__ == "__main__":
    
    # 存放不同图片和json文件的文件夹
    # path = "D:\Programme WorkSpace\Python-Space\ML-design\data"
    path = "D:\Programme WorkSpace\Python-Space\ML-design//test_data"


    # 获取各个文件夹的路径
    folder_path_lst = get_folder_path(path)

    
    processed_img, processed_label = data_store(folder_path_lst)

    print(processed_img.shape)
    print(processed_label.shape)

    np.save(OUTPUT_TEST_IMG, processed_img)
    np.save(OUTPUT_TEST_LABEL, processed_label)



    # 获取每个图像的路径以及对应的标签
    # image_paths_lst, image_labels_lst =  load_samples(folder_path)

    # 将图像编码为RGB三维矩阵形式
    # processed_data= decode_image(image_paths_lst, image_labels_lst)


    # np.save(OUTPUT_TRAIN_DATA, processed_data)

    # data = np.load(OUTPUT_TRAIN_DATA, allow_pickle=True)[0]
    
    # data1 = data

    # train = np.vstack((data,data1))

    # print(train.shape)

    

