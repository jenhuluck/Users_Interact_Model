# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:14:19 2020

@author: Jieyun Hu
"""

# This file includes preparing the data, encoding and modeling
# To predict coordinate x and y
# put this file outside the filtered_traces folder
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
# return[0] is 67 vectors which concatenate 64-dim vectors with 3 dim vector representation  of activity
# return[1] is only 3 dim vector            
def gestures_array_to_vector(gestures_dir_array):
    res = []
    res_y = []
    for gestures_dir in gestures_dir_array:
        with open(gestures_dir) as json_file:
            gestures = json.load(json_file)
            get_ui_seq = [*gestures]
            vecs = []
            vecs_y = []
            for ui in get_ui_seq:
                try:
                    vector_64 = ui_to_vector(ui) #add 64 dim vector to activity vector

                    lst_of_activity = gestures.get(ui)
                    if len(lst_of_activity) == 1: #click
                        temp = [0]
                        temp.extend(lst_of_activity[0])
                        temp = np.asarray(temp)
                        vecs_y.append(temp)  # e.g [0, coorX, coorY]
                        vector_67 = np.concatenate((vector_64,temp),axis=0)
                        #print(len(vector_67))
                        vecs.append(vector_67)# 64 dim vector add to the activity vector
                    elif len(lst_of_activity) > 1: #swipe
                        average_of_coor = [float(sum(l))/len(l) for l in zip(*lst_of_activity)]
                        temp = [1]
                        temp.extend(average_of_coor) # e.g [1, coorX, coorY]
                        temp = np.asarray(temp)
                        vecs_y.append(temp)
                        vector_67 = np.concatenate((vector_64,temp),axis=0)
                        vecs.append(vector_67)
                except:
                    pass
        #print(vecs_y)
        #print(vecs)
        res.append(vecs)   
        res_y.append(vecs_y)
    return [res,res_y]       
            
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


trace_length_to_dictionary()
trace_dir_array = find_all_files_in_range(20,54)
vectors_array = gestures_array_to_vector(trace_dir_array)[0]

#print(vectors_array[0])


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
T = 1
D = 67
M = 10
K = 2

i = Input(shape = (T,D))
x = SimpleRNN(M, return_sequences=True)(i)
x = GlobalMaxPool1D()(x)
x = Dropout(0.5)(x)
x = Dense(K,activation = 'relu')(x)
model = Model(i,x)
#model.compile( loss = 'mse', metrics = ['accuracy'], optimizer = Adam(lr = 0.001),)
model.compile(loss = 'mse', optimizer = Adam(lr=0.001), metrics = ['accuracy'],)


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
            y_input = dataset[y_index,:][65:67]
            X.append(x_input)
            y.append(y_input)
            x_index +=1
            y_index +=1
    return array(X), array(y)



X, y = split_dataset_array(vectors_array, T)
print(X.shape)
#print(y.shape)
#print(y[:10])
r = model.fit(X, y, epochs = 100, validation_split = 0.4)


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


test = ['filtered_traces/com.linkedin.android/trace_0/gestures.json']
res = gestures_array_to_vector(test)
#testing for a ui
#The model just predict every next activity to 0
for i in range(len(res[0][0])):
    x = res[0][0][i].reshape(1,1,67)
    yhat = model.predict(x)
    print (yhat)

