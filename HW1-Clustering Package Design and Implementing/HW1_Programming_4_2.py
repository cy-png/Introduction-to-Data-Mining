#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:03:22 2020

@author: cyfile
"""


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

dataset = pd.read_csv('eye.csv', sep = ',')
dataset = np.array(dataset)
k = 2

def find_distance_matrix(data):
    euclid_distance = []
    for i in data:
        distance = []
        for j in data:
            distance.append(np.linalg.norm(i - j) * np.linalg.norm(i - j))
        distance = np.array(distance)
        euclid_distance.append(distance)
    euclid_distance = np.array(euclid_distance)
    return euclid_distance

def inverse_squareform(matrix):
    inv_sqfrm = []
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix[i])):
            inv_sqfrm.append(matrix[i][j])
    inv_sqfrm = np.array(inv_sqfrm)
    return inv_sqfrm

def rbfkernel(gamma, distance):
    return np.exp(-gamma * distance)

def polykernel(data, degree = 2):
    return (1 + np.dot(data, data.T)) ** degree

def main():
    # df=pd.read_csv('./segmentation.data.modified')
    ### Generate Clusters

    df = pd.DataFrame(dataset)
    data = dataset

    ## Visualize the data
    f = plt.figure(1)
    plt.scatter(df[0],df[1])
    f.show()


    ### making kernel matrix
    # distance = find_distance_matrix(data)
    # gamma = 1/(2*np.var(inverse_squareform(distance)))
    # kernel = rbfkernel(gamma, distance)
    kernel = polykernel(data)

    ## Pick 7 Random K means
    random.seed()
    indexes = random.sample(range(0, len(df)), k)

    ### Alpha
    alpha = np.zeros(shape = (len(df), k))

    ### Calculating distance from mean and assigning labels

    for i in range(len(df)):
        min_value = 123213123123123
        min_index = -1
        for j in range(k):
            distance = kernel[i][i] - 2*(kernel[i][indexes[j]]) + kernel[indexes[j]][indexes[j]]
            if distance < min_value:
                min_value = distance
                min_index = j
        alpha[i][min_index] = 1

    # Now iterate and find labels

    for it in range(10):
        class_frequency = pd.DataFrame(alpha)
        cluster_count = [class_frequency[i].value_counts()[1] for i in range(k)]
        new_alpha = np.zeros(shape = (len(df),k))
        for i in range(len(df)):
            min_value = 123213123123123
            min_index = -1
            for j in range(k):
                sum1 = 0
                for m in range(len(df)):
                    sum1 += alpha[m][j] * kernel[i][m]
                sum2 = 0
                for m1 in range(len(df)):
                    for m2 in range(len(df)):
                        sum2 += alpha[m1][j] * alpha[m2][j] * kernel[m1][m2]
                distance = kernel[i][i] - (2 * sum1 / cluster_count[j]) + (sum2/(cluster_count[j] * cluster_count[j]))
                if distance < min_value:
                    min_value = distance
                    min_index = j
            new_alpha[i][min_index] = 1
        alpha = new_alpha[:]

    # C1 = df[i]
    plt.scatter([dataset[i][0] for i in range(len(dataset)) if alpha[i][0] == 1], [dataset[i][1] for i in range(len(dataset)) if alpha[i][0] == 1], color='red', alpha=0.3)
    plt.scatter([dataset[i][0] for i in range(len(dataset)) if alpha[i][1] == 1], [dataset[i][1] for i in range(len(dataset)) if alpha[i][1] == 1], color='blue', alpha=0.3)
    plt.show()
    # plt.scatter(df[0], df[1])

if __name__ == '__main__':
    main()