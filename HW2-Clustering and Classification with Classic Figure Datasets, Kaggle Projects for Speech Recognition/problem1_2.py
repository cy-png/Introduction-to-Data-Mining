#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 00:31:38 2020

@author: cyfile
"""

#PART0 import packages
import numpy as np
import time
import pickle
from pandas.core.frame import DataFrame
import datetime
from sklearn.cluster import KMeans
import pandas as pd
from myown_kmeans import *

#PART1 read data
######
#####
####
###
##
#

beginTime = time.time()
# Define Parameters
batch_size = 100
# learning_rate = 0.005
learning_rate = 0.005
max_steps = 10000


# Read a batch of data
def load_batch(batch):
    with open(batch, "rb") as fo:
        datadict = pickle.load(fo,encoding="latin1")
        data = datadict["data"]
        labels = datadict["labels"]
        # Split into 10000 sets of images, 3 channels
        # data = np.reshape(data, (10000, 3, 32, 32))
        data = np.reshape(data, (10000, 3072))
        labels = np.array(labels)
        return data, labels


# Consolidate data
def load_data(src):
    datasets = {"images_train":[], "labels_train":[],"images_test":[], "labels_test":[]}
    for b in range(1, 6):
        data, labels = load_batch("{0}data_batch_{1}".format(src, b))
        datasets["images_train"].append(data)
        datasets["labels_train"].append(labels)
    datasets["images_train"] = np.concatenate(datasets["images_train"])
    datasets["labels_train"] = np.concatenate(datasets["labels_train"])
    datasets["images_test"], datasets["labels_test"]  = load_batch("{0}test_batch".format(src))
    return datasets

# load data
data_sets = load_data("/Users/cyfile/Documents/Brandeis/Data Mining/HW2/problem1-2/")
#data_sets = load_data("/home/u/fall19/chuyuewu/dm_hm2/problem2/")

images_train = data_sets['images_train']
labels_train = data_sets['labels_train']


#PART2 scikit-learn kmeans
######
#####
####
###
##
#


#model
starttime = datetime.datetime.now()#Caculate time
sample_model = KMeans(n_clusters=10).fit(images_train)#K-Means
endtime = datetime.datetime.now()#Caculate time
scikit_learn_execution_time = (endtime - starttime).seconds
print('scikit-learn execution time:',scikit_learn_execution_time)#Caculate time 429s

#objective function value
cluster=sample_model.labels_
objective_function_value=sample_model.inertia_ #394810072745.4526
print('scikit-learn objective function value:', objective_function_value) 

#accuracy
crosstable_data = {'label':labels_train,
       'cluster':list(cluster)}
df = DataFrame(crosstable_data)
crosstable = pd.crosstab(index = df['label'], columns = df['cluster'])
scikit_accuracy = sum(crosstable.max(axis=0))/sum(crosstable.sum()) #0.22124
print('scikit-learn accuracy:',scikit_accuracy)



#PART3 my kmeans
######
#####
####
###
##
#


#model
starttime_2 = datetime.datetime.now()#Caculate start time
cluster_center,cluster_assign=Kmeans(np.array(images_train),10)
endtime_2 = datetime.datetime.now()#Caculate end time
myown_kmeans_execution_time = (endtime_2 - starttime_2).seconds
print('myown-kmeans execution time:',myown_kmeans_execution_time)#Caculate time

#objective function value
sum_dis = 0
n = np.shape(images_train)[0]
for m in range(10):
    for i in range(n):
        if cluster_assign[i][0] == m:
            sum_dis += sum(np.power(images_train[i]-cluster_center[m],2))
print('objective function value', sum_dis)
#2301568150.6440473

#accuracy
crosstable_data = {'label':labels_train,
       'cluster':list(cluster_assign[:,0])}
df = DataFrame(crosstable_data)
crosstable = pd.crosstab(index = df['label'], columns = df['cluster'])
myown_accuracy = sum(crosstable.max(axis=0))/sum(crosstable.sum()) #0.25666
print('myown accuracy:',myown_accuracy)


#PART4 results_table
######
#####
####
###
##
#

from tabulate import tabulate

results = [['scikit-learn-kmeans', 
            scikit_learn_execution_time, 
            objective_function_value,
            scikit_accuracy], 
           ['myown_kmeans', 
            myown_kmeans_execution_time, 
            sum_dis, 
            myown_accuracy]]

results_table = tabulate(results, 
                         headers=['k-means', 'execution time', 'objective function value', 'accuracy'], 
                         tablefmt='orgtbl')

print(results_table)
