#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:58:03 2020

@author: chuyue
"""

import pandas as pd 
import numpy as np 
import os
import re

os.getcwd() #Get the current working path and see if it is your own target path
os.chdir('/Users/cyfile/Documents/Brandeis/Data Mining/HW0') #if not, change path
path = '/Users/cyfile/Documents/Brandeis/Data Mining/HW0/docs'
os.listdir(path) #See what data is in the target path

datalist = []
for i in os.listdir(path):
    if os.path.splitext(i)[1] == '.txt': #把文件分为文件名和扩展名, 选取后缀为txt的文件加入datalist
        datalist.append(i)
datalist  # 查看datalist

alltext = []
for txt in datalist:
    data_path = os.path.join(path,txt) #path data_path
    x0 = open(data_path)
    f = x0.read().lower()
    alltext.append(f)

# data cleaning
lists_new = []
for j in alltext:
    a = re.sub('[()-,.!\'":;_?\t\n1234567890&$%#@^&*]', "", j)
    lists_new.append(a)
lists_new

list_formal = []
for item in lists_new:
    list_formal.append(list(item.split()))

# count 1 word 
list_oneword = []
for h in list_formal:
    setword = set(h)
    list_oneword.extend(setword)

dic = {}
for each in list_oneword:
        if each in dic:
            dic[each] = dic[each] + 1
        else:
            dic[each] = 1
# print all value
# print(sorted(dic.items(),key=lambda x:x[1],reverse=True))
            
# only print biggest value
HighValue = 0
HighKey = None
for each in dic:
    if dic[each] > HighValue:
        HighValue = dic[each]
        HighKey = each

for each in dic:
    if dic[each] == HighValue:
        Highkey = each
        print(Highkey, HighValue)    

#count 2 words
dic2 = {}
two_words = []
for h in list_formal:
    setword = set(h)
    for x in setword:
        for y in setword:
            if y > x:
                two_words.append((x,y))
                break

for key in two_words.append:

    
combination_2 = []
for words in sss:
    for x in range(len(words)):
        for y in range(len(words)):
            if y > x : 
                combination_2.append((words[x],words[y]))
word_counter_2 = {}
for key in combination_2:
    word_counter_2[key] = word_counter_2.get(key, 0) + 1
for key,value in word_counter_2.items():
    if(value == max(word_counter_2.values())):
        print((key,value))

