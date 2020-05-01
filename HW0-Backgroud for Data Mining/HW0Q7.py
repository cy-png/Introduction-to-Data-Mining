#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:07:36 2020

@author: cyfile
"""



import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/Library/Fonts/Songti.ttc')

a0 = 3.5e-7
a1 = 1.6e-5
a2 = 1.1e-27

Jupiter = [778000,71492,	1.90e27]
Saturn = [1429000,60268,	5.69e26]
Uranus = [2870990,25559,	8.69e25]
Neptune = [4504300,24764,1.02e26]
Earth = [149600,	6378,5.98e24]
Venus = [108200,	6052,4.87e24]
Mars = [227940,	3398,6.42e23]
Mercury = [57910	,2439,3.30e23]
Pluto = [5913520	,1160,1.32e22]

planet = [Jupiter,
Saturn,
Uranus,
Neptune,
Earth,
Venus,
Mars,
Mercury,
Pluto]

i = 0
j = 0
slist = []
smatrix = []
for i in range(9):
    for j in range(9):
        s = np.sqrt(
        a0 * (planet[i][0] - planet[j][0]) ** 2 + 
        a1 * (planet[i][1] - planet[j][1]) ** 2 + 
        a2 * (planet[i][2] - planet[j][2]) ** 2 )
        slist.append(s)
        j = j + 1
    smatrix.append(slist)
    slist = []
    i = i + 1

mymatrix = np.mat(smatrix)  # from list to matrix

print(mymatrix)

def draw():
    #define x axis and y axis
    xLabel = ["Ju","Sa","Ur","Ne","Ea","Ve","Ma","Me","Pl"]
    yLabel = ["Jupiter","Saturn","Uranus","Neptune","Earth","Venus","Mars","Mercury","Pluto"]
 
    #plot data
    fig = plt.figure()
    #define plot 1*1，draw at the first place
    ax = fig.add_subplot(111)
    #define scale
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel, fontproperties=font)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    #style
    im = ax.imshow(mymatrix, cmap=plt.cm.hot_r)
    #add bar in the right
    plt.colorbar(im)
    #add title
    plt.title("Similarity Heat Map", fontproperties=font)
    #show
    plt.show()
 
d = draw()
