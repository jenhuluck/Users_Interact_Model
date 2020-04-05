# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:01:10 2020

@author: Jieyun Hu
"""

#This file is to convert ui in the trace to vectors so that turn the traces to vector sequences
#So that the vectors can be used for RNN model
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import collections 


npy_file_dir = './ui_layout_vectors/ui_vectors.npy'
json_file_dir = './ui_layout_vectors/ui_names.json'
lst = os.listdir("filtered_traces") # For exploring the filtered_traces file
index_search_file = []
with open(json_file_dir) as json_file:
     index_search_file = json.load(json_file)['ui_names']
vectors = np.load(npy_file_dir)     
     
def search_for_index(ui_name):
    full_name = ui_name + '.png'
    return index_search_file.index(full_name)

def search_for_vector(index):
    return vectors[index,:]        

def ui_to_vector(ui_name):
    return vectors[search_for_index(ui_name),:]

def gestures_to_vectors(gestures_dir): 
    with open(gestures_dir) as json_file:
        gestures = json.load(json_file)
        get_ui_seq = [*gestures]
        vecs = []
        for ui in get_ui_seq:
            vecs.append(ui_to_vector(ui))
    return vecs       
              
def gestures_array_to_vector(gestures_dir_array):
    res = []
    for gestures_dir in gestures_dir_array:
        with open(gestures_dir) as json_file:
            gestures = json.load(json_file)
            get_ui_seq = [*gestures]
            vecs = []
            for ui in get_ui_seq:
                try:
                    vecs.append(ui_to_vector(ui))
                except:
                    pass
        res.append(vecs)        
    return res       
            
dict = collections.defaultdict(list) 
def trace_length_to_dictionary():
    for f in lst:
        sublst = os.listdir("filtered_traces/"+f)
        for sub in sublst:
            file_name = "filtered_traces/"+f+"/"+sub+"/gestures.json"
            with open(file_name) as json_file:
                data = json.load(json_file)
                data_len = len(data)
                #dict[data_len].append(f)
                dict[data_len].append(file_name)
               
#trace_length_to_dictionary need to be run ahead of this
#return the list of file names with the same length of trace
def find_files_by_count(count):
    return dict[count]    
    
#given an start and end index, find all files with in the range of trace steps
def find_all_files_in_range(start,end):
    res = []
    for i in range(start,end):
        l = dict[i]
        for each in l:
            res.append(each)
    return res




        
#test_json = 'filtered_traces/com.yummly.android/trace_1/gestures.json'   
#vec = gestures_to_vectors(test_json) #a list , vec[0] is an array
#print(vec)  

from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten, Dropout, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#RNN model
# N = number of samples
# T = sequence length
# D = number of input features
# M = number of hidden units
# K = number of output units
N = 1
T = 5
D = 64
M = 60
K = 64

i = Input(shape = (T,D))
x = GRU(M, return_sequences=True)(i)
x = GlobalMaxPool1D()(x)
x = Dropout(0.5)(x)
x = Dense(K,activation = 'relu')(x)
model = Model(i,x)
model.compile( loss = 'mse', metrics = ['accuracy'], optimizer = Adam(lr = 0.001),)

def split_dataset(dataset, time_step):
    X, y = list(), list()
    len_ = len(dataset)
    x_index = 0
    y_index = T
    while y_index < len_:
        x_input = dataset[x_index:(x_index+time_step), :]
        y_input = dataset[y_index,:]
        X.append(x_input)
        y.append(y_input)
        x_index +=1
        y_index +=1
    return array(X), array(y)

# split the dataset array to X and y
def split_dataset_array(dataset_array, time_step):
    X, y = list(), list()
    for dataset in dataset_array:
        dataset = np.array(dataset)
        len_ = len(dataset)
        x_index = 0
        y_index = T
        while y_index < len_:
            x_input = dataset[x_index:(x_index+time_step), :]
            y_input = dataset[y_index,:]
            X.append(x_input)
            y.append(y_input)
            x_index +=1
            y_index +=1
    return array(X), array(y)


trace_length_to_dictionary()
trace_dir_array = find_all_files_in_range(20,54)
vectors_array = gestures_array_to_vector(trace_dir_array)
X, y = split_dataset_array(vectors_array, T)
#print(X.shape)
#print(y.shape)
r = model.fit(X, y, epochs = 100, validation_split = 0.4)


#vec = np.array(vec)
#a, b = split_dataset(vec, 3)       
#print(a.shape)
#print(b.shape)
        
#r = model.fit(a, b, epochs = 500, validation_split = 0.2)
#lstm loss: 0.0925 - val_loss: 0.1050
#rnn 
    

import matplotlib.pyplot as plt
f1 = plt.figure(1)
plt.title('Loss')
plt.plot(r.history['loss'], label = 'train')
plt.plot(r.history['val_loss'], label = 'test')
plt.legend()
f1.show()

f2 = plt.figure(2)
plt.title('Accuracy')
plt.plot(r.history['acc'], label = 'train')
plt.plot(r.history['val_acc'], label = 'test')
plt.legend()
f2.show()

ui_name = '35'
index = search_for_index(ui_name)

print(search_for_vector(index))