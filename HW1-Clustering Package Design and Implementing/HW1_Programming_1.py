#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:07:21 2020

@author: chuyue
"""

import pandas as pd
import numpy as np
from collections import Counter


'''
class cluster_validity():    
    def __init__(self, cluster, true_class, X1, X2):
        self.cluster = cluster
        self.true_class = true_class
        self.X1 = X1
        self.X2 = X2
''' 

def calculate_distance(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))
  
def Accuracy(cluster, true_class):
    crosstable = pd.crosstab(index = true_class, columns = cluster)
    accuracy = sum(crosstable.max(axis=0))/sum(crosstable.sum())
    print("Accuracy: ",accuracy)


def NMI(cluster, true_class):
    total_number = len(cluster)
    cluster_counter = Counter(cluster)
    true_class_counter = Counter(true_class)
    
    # compute mutual information
    MI = 0
    eps = 1e-16 # to avoid log 0

    for i in cluster_counter:
        for j in true_class_counter:
            count = 0
            for k in range(total_number):
                if cluster[k] == i and true_class[k] == j:
                    count += 1
            Pi = cluster_counter[i] / total_number
            Pj = true_class_counter[j] / total_number
            Pij = count / total_number
            MI += Pij * np.log(Pij /(Pi * Pj) + eps)
    
    # normorlize mutual information
    entropy_cluster = 0
    entropy_true_class = 0
    for i in cluster_counter:
        Pi = cluster_counter[i] / total_number
        entropy_cluster -= Pi * np.log(Pi + eps)
    for j in true_class_counter:
        Pj = true_class_counter[j] / total_number
        entropy_true_class -= Pj * np.log(Pj + eps)

    NMI = MI/((entropy_cluster + entropy_true_class)/2.0)
    print('Normalized Mutual Information: ',NMI)


def NRI(cluster, true_class):
    total_number = len(cluster)
    M = total_number * (total_number-1)/2.0
    m = 0
    m1 = 0
    m2 = 0
    for i in range(total_number):
        for j in range(i + 1, total_number):
            if true_class[i] == true_class[j] and cluster[i] == cluster[j]:
                m += 1
    for i in range(total_number):
        for j in range(i + 1, total_number):
            if cluster[i] == cluster[j]:
                m1 += 1
    for i in range(total_number):
        for j in range(i + 1, total_number):
            if true_class[i] == true_class[j]:
                m2 += 1
    NRI = (m - m1*m2/M)/(m1/2 + m2/2 - m1*m2/M)
    print('Normalized Rand Index', NRI)


def Silhouette_Index(cluster, dataset):
    total_number = len(cluster)
    
    #calculate ai
    a_list = []
    for i in range(total_number):
        ai = 0
        a = 0 
        counta = 0
        for j in range(total_number):
            if i != j and cluster[i] == cluster[j]:
                counta += 1
                ai = calculate_distance(dataset[i],dataset[j])
                a += ai
        a_list.append(a/counta)
    
    #calculate bi
    cluster_counter = Counter(cluster)
    b_list = []
    for i in range(total_number):
        b_avg = []
        for k in cluster_counter:
            countb = 0
            bi = 0
            b = 0
            if cluster[i] != k:
                for j in range(total_number):
                    if i!= j and cluster[j] == k:
                        countb += 1
                        bi = calculate_distance(dataset[i],dataset[j])
                        b += bi
                b_avg.append(b/countb)
        b_list.append(min(b_avg))

    #calculate si
    s_list = []
    for i in range(total_number):
        if b_list[i] == a_list[i]:
            si = 0
        elif b_list[i] != a_list[i]:
            si = (b_list[i] - a_list[i])/max(a_list[i],b_list[i]) 
        s_list.append(si)
    print('Silhouette Index: ',np.mean(s_list))       


'''
#import data
dataset = pd.read_csv('programming1.csv', sep = ',')
print(dataset)

#cluster
kmeans = KMeans(n_clusters=3).fit(dataset['score'].values.reshape(-1,1))
labels = kmeans.labels_
dataset['clusters']=labels

cluster = dataset['clusters']
true_class = dataset['Ture_class']
X1 = dataset['score']
X2 = dataset['score']


Accuracy(cluster, true_class)
NMI(cluster, true_class)
NRI(cluster, true_class)
a.Silhouette_Index(cluster, X1, X2)
'''











       