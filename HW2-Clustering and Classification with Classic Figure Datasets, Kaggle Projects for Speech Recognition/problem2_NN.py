#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:15:56 2020

@author: cyfile
"""
#PART0 import packages

import numpy as np
import struct
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
import datetime
from pandas.core.frame import DataFrame



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



'''
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
'''

train_images = load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()

train_images_list = []
for i in train_images:
    train_images_row = []
    for j in i:
        train_images_row.extend(j)
    train_images_list.append(train_images_row)
train_images_dataframe=DataFrame(train_images_list)

test_images_list = []
for i in test_images:
    test_images_row = []
    for j in i:
        test_images_row.extend(j)
    test_images_list.append(test_images_row)
test_images_dataframe=DataFrame(test_images_list)

#PART2 build model
######
#####
####
###
##
#

X_train = train_images_dataframe
Y_train = train_labels
X_test = test_images_dataframe
Y_test = test_labels

X_train = scale(X_train)
X_test = scale(X_test)



#train model nn
nodes_scores = []

starttime = datetime.datetime.now()#Caculate time


for nodes in range(1,101):
    mlp = MLPClassifier(hidden_layer_sizes=(nodes,), max_iter=10, alpha=1e-4, 
                    activation = 'identity',solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)
    scores = cross_val_score(mlp, X_train, Y_train, cv=2, scoring='accuracy')
    print(nodes, scores.mean())
    nodes_scores.append(scores.mean())


endtime = datetime.datetime.now()#Caculate time
nn_execution_time = (endtime - starttime).seconds
print('nn_execution_time:',nn_execution_time)#Caculate time



'''
nn_execution_time: 1124
1 accuracy 0.33503333333333335
2 accuracy 0.60285
3 accuracy 0.7842666666666667
4 accuracy 0.8479666666666668
5 accuracy 0.8734
6 accuracy 0.8731833333333333
7 accuracy 0.8856333333333334
8 accuracy 0.88785
9 accuracy 0.8891166666666667
10 accuracy 0.8917333333333333
11 accuracy 0.897
12 accuracy 0.8852
13 accuracy 0.8857666666666666
14 accuracy 0.8927166666666666
15 accuracy 0.89225
16 accuracy 0.8894166666666666
17 accuracy 0.8763666666666666
18 accuracy 0.89165
19 accuracy 0.8802333333333334
20 accuracy 0.8840333333333333
21 accuracy 0.8744666666666667
22 accuracy 0.8874
23 accuracy 0.88475
24 accuracy 0.8655333333333333
25 accuracy 0.8764000000000001
26 accuracy 0.8760333333333333
27 accuracy 0.8884666666666667
28 accuracy 0.86485
29 accuracy 0.8901
30 accuracy 0.86385
31 accuracy 0.8739666666666667
32 accuracy 0.8722
33 accuracy 0.8806166666666667
34 accuracy 0.8748
35 accuracy 0.8623833333333333
36 accuracy 0.8426166666666667
37 accuracy 0.8598166666666667
38 accuracy 0.8687333333333334
39 accuracy 0.8784333333333334
40 accuracy 0.8686166666666666
41 accuracy 0.8748833333333333
42 accuracy 0.8355166666666667
43 accuracy 0.8706
44 accuracy 0.8673166666666667
45 accuracy 0.8015000000000001
46 accuracy 0.8617333333333334
47 accuracy 0.8600833333333333
48 accuracy 0.84765
49 accuracy 0.8696
50 accuracy 0.8567333333333333
51 accuracy 0.8481666666666667
52 accuracy 0.8553666666666666
53 accuracy 0.8783833333333333
54 accuracy 0.8640666666666666
55 accuracy 0.8353666666666667
56 accuracy 0.8635
57 accuracy 0.8486333333333334
58 accuracy 0.8644166666666666
59 accuracy 0.86695
60 accuracy 0.8675333333333333
61 accuracy 0.8546666666666667
62 accuracy 0.85495
63 accuracy 0.8745833333333333
64 accuracy 0.8638833333333333
65 accuracy 0.8710666666666667
66 accuracy 0.8654
67 accuracy 0.8627333333333334
68 accuracy 0.8581166666666666
69 accuracy 0.8735666666666666
70 accuracy 0.8537166666666667
71 accuracy 0.8614166666666667
72 accuracy 0.8305833333333333
73 accuracy 0.8141166666666667
74 accuracy 0.8463666666666667
75 accuracy 0.8458666666666668
76 accuracy 0.8384
77 accuracy 0.85305
78 accuracy 0.8487666666666667
79 accuracy 0.8523000000000001
80 accuracy 0.8530666666666666
81 accuracy 0.8753833333333334
82 accuracy 0.8597333333333333
83 accuracy 0.85245
84 accuracy 0.8643166666666666
85 accuracy 0.8276333333333333
86 accuracy 0.8494166666666667
87 accuracy 0.8584
88 accuracy 0.8539833333333333
89 accuracy 0.87965
90 accuracy 0.86895
91 accuracy 0.8466166666666667
92 accuracy 0.84745
93 accuracy 0.8583333333333334
94 accuracy 0.8433333333333333
95 accuracy 0.85855
96 accuracy 0.84785
97 accuracy 0.8549
98 accuracy 0.8592666666666666
99 accuracy 0.8497333333333333
100 accuracy 0.8440166666666666
'''

mlp = MLPClassifier(hidden_layer_sizes=(11,), max_iter=100, alpha=1e-4, 
                activation = 'identity',solver='sgd', verbose=10, random_state=1,
                learning_rate_init=.1)
mlp.fit(X_train, Y_train)
print(
      "Training set score: %f" % mlp.score(X_train, Y_train), 
      "Test set score: %f" % mlp.score(X_test, Y_test))


