#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:08:17 2020

@author: cyfile
"""

#PART0 import packages
from sklearn.datasets import fetch_lfw_people
from pandas.core.frame import DataFrame
from sklearn.cluster import KMeans
from myown_kmeans import *
import pandas as pd
import datetime



#PART1 read data
######
#####
####
###
##
#
lfw_people = fetch_lfw_people(min_faces_per_person=99, resize=0.4)
lfw_people_data = lfw_people['data']
lfw_people_target = lfw_people['target']
lfw_people_target_names = lfw_people['target_names']


#PART2 scikit-learn kmeans
######
#####
####
###
##
#

#model
starttime = datetime.datetime.now()#Caculate time
lfw_model = KMeans(n_clusters=5).fit(lfw_people_data)#K-Means
endtime = datetime.datetime.now()#Caculate time
scikit_learn_execution_time = (endtime - starttime).seconds
print('scikit-learn execution time:',scikit_learn_execution_time)#Caculate time

#objective function value
lfw_cluster=lfw_model.labels_
scikit_learn_objective_function_value=lfw_model.inertia_ #2226310261.650631
print('scikit-learn objective function:',scikit_learn_objective_function_value)

#accuracy
crosstable_data = {'label':list(lfw_people_target),
       'cluster':list(lfw_cluster)}
df = DataFrame(crosstable_data)
crosstable = pd.crosstab(index = df['label'], columns = df['cluster'])
scikit_accuracy = sum(crosstable.max(axis=0))/sum(crosstable.sum()) #0.277
print('scikit-learn accuracy:',scikit_accuracy)


#PART3 my kmeans
######
#####
####
###
##
#

#model
starttime = datetime.datetime.now()#Caculate start time
cluster_center,cluster_assign=Kmeans(array(lfw_people_data),5)
endtime = datetime.datetime.now()#Caculate end time
myown_kmeans_execution_time = (endtime - starttime).seconds
print('myown-kmeans execution time:',myown_kmeans_execution_time)#Caculate time

#objective function value
starttime = datetime.datetime.now()#Caculate start time
sum_dis = 0
n = np.shape(lfw_people_data)[0]
for m in range(5):
    for i in range(n):
        if cluster_assign[i][0] == m:
            sum_dis += sum(power(lfw_people_data[i]-cluster_center[m],2))
print('objective function value',sum_dis)
#2236008838.4331517
endtime = datetime.datetime.now()#Caculate end time
objective_function_execution_time = (endtime - starttime).seconds
print('objective function execution time:',objective_function_execution_time )#Caculate time

#accuracy
crosstable_data = {'label':lfw_people_target,
       'cluster':list(cluster_assign[:,0])}
df = DataFrame(crosstable_data)
crosstable = pd.crosstab(index = df['label'], columns = df['cluster'])
myown_accuracy = sum(crosstable.max(axis=0))/sum(crosstable.sum()) #0.25666
print('myown accuracy:',myown_accuracy)#0.47543859649122805


#PART4 results
######
#####
####
###
##
#
from tabulate import tabulate

results = [['scikit-learn-kmeans', 
            scikit_learn_execution_time, 
            scikit_learn_objective_function_value,
            scikit_accuracy], 
           ['myown_kmeans', 
            myown_kmeans_execution_time, 
            sum_dis, 
            myown_accuracy]]

results_table = tabulate(results, 
                         headers=['k-means', 'execution time', 'objective function value', 'accuracy'], 
                         tablefmt='orgtbl')

print(results_table)