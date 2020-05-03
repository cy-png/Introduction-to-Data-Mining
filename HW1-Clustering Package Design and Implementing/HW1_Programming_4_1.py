#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:53:51 2020

@author: cyfile
"""



import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


dataset = pd.read_csv('eye.csv', sep = ',')
dataset = np.array(dataset)

cutoff = 0.98
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

def KMeans_implementation(data):
    alpha = np.zeros(shape = (len(data), k))

    random.seed()
    indexes = random.sample(range(0, len(data)), k)
    means = data[indexes]

    for i in range(15):
        new_alpha = np.zeros(shape = (len(data),k))
        for i in range(len(data)):
            min_value = 123213123123123
            min_index = -1
            for j in range(k):
                distance = np.linalg.norm(data[i] - means[j])**2
                if distance < min_value:
                    min_value = distance
                    min_index = j
            new_alpha[i][min_index] = 1
        alpha = new_alpha[:]
        class_frequency = pd.DataFrame(alpha)    
        cluster_count = [class_frequency[i].value_counts()[1] for i in range(k)]
        for j in range(k):
            means[j] = 0
            for i in range(len(data)):
                means[j] += (alpha[i][j] * data[i])
            means[j] = means[j] / float(cluster_count[j])

    labels = []
    for i in range(len(alpha)):
        for j in range(k):
            if alpha[i][j] == 1:
                labels.append(j)
    return labels

def main():

    df = pd.DataFrame(dataset)
    data = dataset

    # Visualize the data
    f = plt.figure(1)
    plt.scatter(df[0],df[1])
    f.show()

    # Making kernel matrix
    distance = find_distance_matrix(data)
    gamma = 1/(2*np.var(inverse_squareform(distance)))
    kernel = rbfkernel(gamma, distance)

    # Calculating weight matrix, and using cutoff to indicate far away points
    W = kernel[:]
    for i in range(len(kernel)):
        for j in range(len(kernel)):
            if kernel[i][j] < cutoff:
                W[i][j] = 0

    # Making degree matrix
    D = np.zeros(shape = (len(W), len(W)))
    for i in range(len(W)):
        D[i][i] = np.sum(W[i])           

    # Calculate Laplacian Matrix
    L = D - W

    # Find eigen vectors of Laplacian Matrix
    eigen_values, eigen_vectors = np.linalg.eig(L)
    indexes = eigen_values.argsort()[::1]
    second_vector = eigen_vectors[:, indexes[1:k]]

    # Use K-means to find labels
    label = KMeans_implementation(second_vector)
    print(label)

    g = plt.figure(2)
    plt.scatter([dataset[i][0] for i in range(len(dataset)) if label[i] == 1], [dataset[i][1] for i in range(len(dataset)) if label[i] == 1], color='red', alpha=0.5)
    plt.scatter([dataset[i][0] for i in range(len(dataset)) if label[i] == 0], [dataset[i][1] for i in range(len(dataset)) if label[i] == 0], color='blue', alpha=0.5)
    g.show()


if __name__ == '__main__':
    main()