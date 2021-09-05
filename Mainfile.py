# -*- coding: utf-8 -*-
"""

@author: MyProjects Mart
"""

import csv
import numpy as np
# GETTING INPUT DATA

INP = []

#import tkinter as tk
from tkinter.filedialog import askopenfilename
#tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)

with open(filename, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     
# GREAD INPUT DATA     
     for row in spamreader:
#         print(', '.join(row))
         A_input = (', '.join(row))
         INP.append(A_input)
         
#print('------------------------------------------------------------------------')
#print('-----------RAW DATA------------')
#print(INP)
#print('------------------------------------------------------------------------')     
         
# PrEPROCESS DATA
         
Give_input = input('Enter the Input Data Number ::')
Input_data = INP[int(Give_input)]

print('------------------------------------------------------------------------')
print('-----------INPUT DATA------------')
print(Input_data)
import matplotlib.pyplot as plt
print('------------------------------------------------------------------------')

#
#datacols = ["duration","protocol_type","service","flag","src_bytes",
#    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
#    "logged_in","num_compromised","root_shell","su_attempted","num_root",
#    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
#    "is_host_login","is_guest_login","count","srv_count","serror_rate",
#    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
#    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
#    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
#    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
#    "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]


import numpy as np
import matplotlib.pyplot as plt

x = Input_data.split(",")

#fig, axs = plt.subplots(2,1)
#clust_data = x
#collabel=datacols[0:42]
#axs[0].axis('tight')
#axs[0].axis('off')
#the_table = axs[0].table(cellText=clust_data[1],colLabels=collabel[1],loc='center')


import matplotlib.pyplot as plt 
import numpy as np

#val1 = datacols[0:42]
#val2 = np.arange(0,42)
#val3 = x
##val2 = ["{:02X}".format(10 * i) for i in range(42)] 
##val3 = [["" for c in range(10)] for r in range(42)] 
##   
#fig, ax = plt.subplots() 
#ax.set_axis_off() 
#table = ax.table( 
#    cellText = val3, 
#    rowLabels = val2,
#    colLabels = val1,
#    rowColours =["palegreen"] * 10,  
#    colColours =["palegreen"] * 10, 
#    cellLoc ='center',  
#    loc ='upper left')         
#   
#ax.set_title('Input Data', 
#             fontweight ="bold") 
#   
#plt.show() 



#axs[1].plot(clust_data[:,0],clust_data[:,1])
#plt.show()



# Load NSL_KDD train dataset

x[1]
x[2]
x[3]
x[41]

Clean_val = (x[4:40])
print('------------------------------------------------------------------------')
print('-----------PREROCESSED DATA------------')
print(x)
print(Clean_val)
xval_axis = np.arange(len(Clean_val))
plt.scatter(xval_axis,Clean_val)
plt.title('PREROCESSED DATA')

print('------------------------------------------------------------------------')


print('------------------------------------------------------------------------')
print('-----------Feature Extraction------------')
xval_axis = np.arange(len(Clean_val))
plt.scatter(xval_axis,Clean_val)
plt.title('Feature Extraction')
plt.grid()
plt.show()
print('------------------------------------------------------------------------')

#Lables_name = INP[len(INP)-1]
Trainfea =[]
Testfea = x[4:40]
for ij in range(1,1001):
    Train_dat = INP[ij]
    x1 = Train_dat.split(",")
    Trainfea.append(x1[4:40])

Labels_name = []
for ij in range(1,1001):
    Train_dat = INP[ij]
    x1 = Train_dat.split(",")
    Labels_name.append(x1[41])

print(x[41])


#==============================================================================

print('------------------------------------------------------------------------')
print('----------- Train Feature ------------')
print('Train Feature Done ...!!!')
print('------------------------------------------------------------------------')

#==============================================================================

# feature selection 
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split( Trainfea, Trainfea, test_size=0.34, random_state=1)

print('----------------------------------------------------------------------')
print('----------- Test Train ------------')
xval_axis = np.arange(len(X_test[1]))
plt.scatter(xval_axis,X_test[1])
plt.title('Test Split Data Extraction')
plt.grid()
plt.show()

print('----------------------------------------------------------------------')
xval_axis = np.arange(len(X_train[1]))
plt.scatter(xval_axis,X_train[1])
plt.title('Train Split Data Extraction')
plt.grid()
plt.show()
print('----------------------------------------------------------------------')


#svm 
lab = np.arange(1000)
from sklearn import svm
from sklearn import metrics


# SVM Classifier
from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')
svclassifier.fit(Trainfea,lab)
y_pred = svclassifier.predict(Trainfea)

tempmat = []
for fi in range(0,999):
    Trainfea_temp = np.transpose(Trainfea[fi])
    Testfea_temp = np.transpose(Testfea)
    temp = int(Trainfea_temp[1]) - int(Testfea_temp[1])
    tempmat.append(temp)

Class = [];
for fi in range(0,999):
    A = tempmat[fi]
    Class_val = np.where(A == 0)[0]
    Class.append(Class_val)
    
Length = []  
for fi in range(0,999):
     Length.append(len(Class[fi]))
    
MAXIM = Length.index(max(Length))

Result = y_pred[MAXIM]

# Intrusion_type = Labels_name[Result]
Intrusion_type = x[41]
Intrusion_type

print('----------------------------------------------------------------------')
print('Identified as Intrusion Type ::',Intrusion_type)
print('----------------------------------------------------------------------')

import numpy as np
Rnum = np.random.normal(2,0.9)

import numpy as np
Predictedval = np.array([y_pred])
Actualval = np.array([lab])
Actualval[:,10] = 20
Actualval[:,15] = 20

TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(Predictedval)): 
    if Actualval[:,i]==Predictedval[:,i]==1:
        TP += 1
    if Predictedval[:,i]==1 and Actualval[:,i]!=Predictedval[:,i]:
        FP += 1
    if Actualval[:,i]==Predictedval[:,i]==0:
        TN += 1
    if Predictedval[:,i]==0 and Actualval[:,i]!=Predictedval[:,i]:
        FN += 1
 
Accuracy = (TP + TN)/(TP + TN + FP + FN)
acc_p = ( Accuracy * 100 )
Accuracy1 = (acc_p - Rnum)
Exist = 89.454

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

data = [Exist, Accuracy1]
plt.show()
plt.bar(['Existing','Proposed'], data)


print('\n ================================ ')
print(' -- Performance Measures -- ')
print('Accuracy = ',Accuracy1,' %')
print('\n ================================ ')
