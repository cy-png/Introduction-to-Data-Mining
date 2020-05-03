#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 23:15:26 2020

@author: cyfile
"""

#PART0 import packages

import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import datetime
from pandas.core.frame import DataFrame
import pandas as pd
from myown_kmeans import *

#PART1 read data
######
#####
####
###
##
#

# train images
train_images_idx3_ubyte_file = 'train-images-idx3-ubyte'
# train label
train_labels_idx1_ubyte_file = 'train-labels-idx1-ubyte'

# test images
test_images_idx3_ubyte_file = 't10k-images-idx3-ubyte'
# test label
test_labels_idx1_ubyte_file = 't10k-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    General functions for parsing idx3 files
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)



def run():
    train_images = load_train_images()
    train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    for i in range(10):
        print(train_labels[i])
        plt.imshow(train_images[i], cmap='gray')
        plt.show()
    print('done')

if __name__ == '__main__':
    run()


train_images = load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()


#PART2 scikit-learn kmeans
######
#####
####
###
##
#

#60000-row dataframe
train_images_list = []
for i in train_images:
    train_images_row = []
    for j in i:
        train_images_row.extend(j)
    train_images_list.append(train_images_row)

train_images_dataframe=DataFrame(train_images_list)

#build model
starttime = datetime.datetime.now()#Caculate time
scikit_model = KMeans(n_clusters=10).fit(train_images_dataframe)#K-Means
endtime = datetime.datetime.now()#Caculate time
scikit_learn_kmeans_execution_time = (endtime - starttime).seconds
print('scikit-learn kmeans execution time',scikit_learn_kmeans_execution_time)#Caculate time 103s

#objective function and accuracy
cluster=scikit_model.labels_
objective_function_value=scikit_model.inertia_ #152992612959.03403
print('objective function value', objective_function_value)

train_labels_int = []
for i in train_labels:
    train_labels_int.append(int(i))


crosstable_data = {'label':train_labels_int,
       'cluster':list(cluster)}
df = DataFrame(crosstable_data)
crosstable = pd.crosstab(index = df['label'], columns = df['cluster'])
accuracy = sum(crosstable.max(axis=0))/sum(crosstable.sum())
print('scikit-learn accuracy', accuracy)#0.5907


#PART3 my kmeans
######
#####
####
###
##
#

#model
starttime = datetime.datetime.now()#Caculate start time
cluster_center,cluster_assign=Kmeans(array(train_images_dataframe),10)
endtime = datetime.datetime.now()#Caculate end time
myown_kmeans_execution_time = (endtime - starttime).seconds
print('myown-kmeans sexecution time',myown_kmeans_execution_time)#Caculate time 1985s

#objective function valule
starttime = datetime.datetime.now()#Caculate start time
sum_dis = 0
n = np.shape(train_images_dataframe)[0]
for m in range(10):
    for i in range(n):
        if cluster_assign[i][0] == m:
            sum_dis += sum(power(array(train_images_dataframe)[i]-cluster_center[m],2))
print('objective function value:',sum_dis)
#695738146.0963647
endtime = datetime.datetime.now()#Caculate end time
print((endtime - starttime).seconds)#Caculate time

#accuracy
crosstable_data = {'label':train_labels_int,
       'cluster':list(cluster_assign[:,0])}
df = DataFrame(crosstable_data)
crosstable = pd.crosstab(index = df['label'], columns = df['cluster'])
myown_accuracy = sum(crosstable.max(axis=0))/sum(crosstable.sum()) #0.25666
print('myown accuracy:',myown_accuracy)


#PART4 results
######
#####
####
###
##
#
from tabulate import tabulate

results = [['scikit-learn-kmeans', 
            scikit_learn_kmeans_execution_time, 
            objective_function_value,
            accuracy], 
           ['myown_kmeans', 
            myown_kmeans_execution_time, 
            sum_dis, 
            myown_accuracy]]

results_table = tabulate(results, 
                         headers=['k-means', 'execution time', 'objective function value', 'accuracy'], 
                         tablefmt='orgtbl')

print(results_table)
