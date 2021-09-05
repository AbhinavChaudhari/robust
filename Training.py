# -*- coding: utf-8 -*-
"""

@author: Myprojects Mart
"""


import csv

# GETTING INPUT DATA

INP = []
Trainfea = []
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
for rown in range(0,len(INP)):
    Give_input = rown
    print(Give_input)
    Input_data = INP[Give_input]
    x = Input_data.split(",")
    Clean_val = (x[4:40])
    Trainfea.append(Clean_val[4:40])
#print('------------------------------------------------------------------------')
#print('-----------RAW DATA------------')
#print(INP)
#print('------------------------------------------------------------------------')     
         
# PrEPROCESS DATA
         
        
import pickle
with open('Trainfea1.pickle', 'wb') as f:
    pickle.dump(Trainfea, f)