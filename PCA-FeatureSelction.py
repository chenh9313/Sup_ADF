#!/usr/bin/env python
# coding: utf-8

#Author: Huan Chen
#Date: 02/27/2024

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler

#read testdata
data = pd.read_table('balanced_test_148data.txt', sep=" ")
#feature_name = data.columns.tolist()[2:]
X = data[data.columns.tolist()[2:]]#get all the feature name
X = X.to_numpy()

#Using StandardScaler and PCA
std_slc = StandardScaler()
X_std = std_slc.fit_transform(X)
print(X_std.shape)

#If 0 < n_components < 1, how much precent variance could be explained
#pca = decomposition.PCA(n_components=4) #you decide how many demention you want
pca = decomposition.PCA(n_components=0.85)
X_std_pca = pca.fit_transform(X_std)

temp = list(X_std_pca)

with open('res_raw_PCAfeatureselection.txt', 'w') as f:
    for line in temp:
        f.write(f"{line}\n")

#check each PC explain how much precent of the variance
print(pca.explained_variance_ratio_)

#some command on Linux
#orgnized the data structure
#remove [] for res_raw_PCAfeatureselection.txt
awk '{print $1","$2}' res_raw_PCAfeatureselection.txt > res_PCAfeatureselection.txt
add "TPMPCA1,TPMPCA2"	to res_PCAfeatureselection.txt
awk -F"," '{print NF}' res_PCAfeatureselection.txt | uniq

awk '{print $1","$2}' balanced_test_148data.txt > temp
paste -d "," temp res_PCAfeatureselection.txt > balanced_test_148data_PCAreduced.txt
rm temp

#organize single-cell infomation "arabidopsis_thaliana.marker_fd.csv"
awk -F"," '{print $1","$8","$9","$12","$13}' arabidopsis_thaliana.marker_fd.csv > sc_marker_info.csv
sed -i 's/inf/NA/g' sc_marker_info.csv #$2, avg_log2FC some value is inf
awk -F"," '{print $3}' sc_marker_info.csv | sort | uniq | awk '{ print length, $0 }' | sort -n -r -s | cut -d" " -f2-


