#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:47:29 2020

@author: cyfile
"""

#PART0 import packages
######
#####
####
###
##
#
import pandas as pd
import numpy as np
import random
import librosa #for reading audio file
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.core.frame import DataFrame
from numpy import argmax

# Libraries for Classification and building Models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical 


#PART1 read data
######
#####
####
###
##
#

##import csv data
UrbanSound8K_csv = pd.read_csv("UrbanSound8K.csv")
UrbanSound8K_csv.head()

##explore wav data
#At first, select five files random to see their patterns
five_random_sample_index = []
for i in range(5):
    five_random_sample_index.append(random.randint(0,8732))

slice_file_name = np.array(UrbanSound8K_csv["slice_file_name"])
fold = np.array(UrbanSound8K_csv["fold"])
class_name = np.array(UrbanSound8K_csv["class"])
class_ID = np.array(UrbanSound8K_csv["classID"])

for j in five_random_sample_index:
    path = '/Users/cyfile/Documents/Brandeis/Data Mining/HW2/problem4/urbansound8k/fold' + str(fold[j]) + '/' + slice_file_name[j]
    data, sampling_rate = librosa.load(path)
    plt.figure(figsize=(10, 5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(class_name[j])

#Then, we use librosa.mel_spectogram function to extract features of data
#for one data:
file_name_1 = '/Users/cyfile/Documents/Brandeis/Data Mining/HW2/problem4/urbansound8k/fold' + str(fold[0]) + '/' + slice_file_name[0]
dat1, sampling_rate1 = librosa.load(file_name_1)
array_1 = librosa.feature.melspectrogram(y=dat1, sr=sampling_rate1)
array_1.shape

#for all the data:
feature = []
label = []

def parser():
    # Function to load files and extract features
    for i in range(8732):
        file_name = '/Users/cyfile/Documents/Brandeis/Data Mining/HW2/problem4/urbansound8k/fold' + str(fold[i]) + '/' + slice_file_name[i]
        dat, sampling_rate = librosa.load(file_name, res_type='kaiser_fast') #kaiser_fast is a technique used for faster extraction
        # We extract mfcc feature from data
        mels = np.mean(librosa.feature.melspectrogram(y=dat, sr=sampling_rate).T,axis=0)        
        feature.append(mels)
        label.append(class_ID[i])
    return [feature, label]

dataset = (np.array(parser())).transpose()
dataset.shape

X = np.empty([8732, 128])
for i in range(8732):
    X[i] = dataset[:,0][i]
Y = to_categorical(dataset[:,1])

#split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
X_train = X_train.reshape(6549, 16, 8, 1)
X_test = X_test.reshape(2183, 16, 8, 1)

input_dim = (16, 8, 1)


#PART2 build model
######
#####
####
###
##
#

'''
Model 1:
    CNN 2D with 64 units and tanh activation.
    MaxPool2D with 2*2 window.
    CNN 2D with 128 units and tanh activation.
    MaxPool2D with 2*2 window.
    Dropout Layer with 0.2 drop probability.
    DL with 1024 units and tanh activation.
    DL 10 units with softmax activation.
    Adam optimizer with categorical_crossentropy loss function.
90 epochs have been used.
'''
model = Sequential()
model.add(Conv2D(64, (3, 3), padding = "same", activation = "tanh", input_shape = input_dim))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding = "same", activation = "tanh"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation = "tanh"))
model.add(Dense(10, activation = "softmax"))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = 100, batch_size = 50, validation_data = (X_test, Y_test))
model.summary()

predictions = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print(score)

preds = np.argmax(predictions, axis = 1)

result = pd.DataFrame(preds)
result.to_csv("Results.csv")


#PART3 Results Analysis
######
#####
####
###
##
#
test_label = []
for i in Y_test:
    test_label.append(int(max(i)))

test_label = argmax(Y_test,axis = 1)
crosstable_data = {'label':test_label,
       'class':list(preds)}
df = DataFrame(crosstable_data)

#wrong prediction
path = '/Users/cyfile/Documents/Brandeis/Data Mining/HW2/problem4/urbansound8k/fold' + str(fold[4]) + '/' + slice_file_name[4]
data, sampling_rate = librosa.load(path)
plt.figure(figsize=(10, 5))
D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title(class_name[4])


path = '/Users/cyfile/Documents/Brandeis/Data Mining/HW2/problem4/urbansound8k/fold' + str(fold[12]) + '/' + slice_file_name[12]
data, sampling_rate = librosa.load(path)
plt.figure(figsize=(10, 5))
D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title(class_name[12])


#successful prediction
path = '/Users/cyfile/Documents/Brandeis/Data Mining/HW2/problem4/urbansound8k/fold' + str(fold[443]) + '/' + slice_file_name[443]
data, sampling_rate = librosa.load(path)
plt.figure(figsize=(10, 5))
D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title(class_name[443])


path = '/Users/cyfile/Documents/Brandeis/Data Mining/HW2/problem4/urbansound8k/fold' + str(fold[1081]) + '/' + slice_file_name[1081]
data, sampling_rate = librosa.load(path)
plt.figure(figsize=(10, 5))
D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title(class_name[1081])