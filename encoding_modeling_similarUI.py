# This file includes preparing the data, encoding and modeling
# To predict two type of activity. Click = 0 , Swipe = 1.
# put this file outside the filtered_traces folder
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import collections 

# find similar UI
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

# ui vectors
vectors = np.load('ui_layout_vectors/ui_vectors.npy')
ui_vectors = vectors.tolist()

# Opening JSON file and loading the data 
# into the variable data 
with open('ui_layout_vectors/ui_names.json') as json_file: 
    data_names = json.load(json_file) 

# ui names dataframe
names = data_names['ui_names']
names = np.array(names)
ui_names = pd.DataFrame(names, columns = ['name'])

# ui details dataframe
ui_details = pd.read_csv('ui_details.csv')
ui_details.columns = ['ui_num','package_name','interaction_trace','ui_trace_num']

# get query ui's vector
def get_query_ui_vector(ui_name):
    # get index of file
    idx = ui_names.index[ui_names['name']==ui_name]
    #idx = ui_names.index[ui_names['name']=='11374.png']
    #print(idx)
    # query ui's vector
    ui_query = vectors[idx,:]
    
    #print(ui_query)
    return ui_query

# find similar/retrieved ui names
def find_similar_ui_names(ui_query, k):
    # using nearest neighbors to find similar UIs
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(ui_vectors)
    # indices of similar/retrieved UIs
    retrieved_idx = neigh.kneighbors(ui_query, return_distance=False)
    #print(retrieved_idx)

    # retrieved ui names
    retrieved_ui = []
    for index in retrieved_idx[0]:
        retrieved_ui.append(names[index])
    
    #print(retrieved_ui)
    return retrieved_ui

# get package names of chosen UIs
def get_package_names(uis):
    uipackages = []
    for ui in uis:
        uinum = int(re.search(r'\d+', ui).group())
        package = ui_details[ui_details['ui_num']==uinum]['package_name'].values
        trace = ui_details[ui_details['ui_num']==uinum]['interaction_trace'].values
        # exclude not existing UIs and packages
        if package:
            package_gesturefile = 'filtered_traces/'+package[0]+'/trace_'+str(trace[0])+'/gestures.json'
            uipackages.append(package_gesturefile)
    
    #print(uipackages)
    return uipackages

#ui_query = get_query_ui_vector('36043.png') #lists
#ui_query = get_query_ui_vector('37530.png') #lists with larger graph
#ui_query = get_query_ui_vector('36048.png') #login screen
ui_query = get_query_ui_vector('11374.png') #calendar
#ui_query = get_query_ui_vector('21506.png') #image grids
ui_retrieved = find_similar_ui_names(ui_query, 500)
uipackages = get_package_names(ui_retrieved)


# encoding_modeling
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

#find the 64-dim vector
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


# Given a list of paths of file directory.
# return[0] is 67 vectors which concatenate 64-dim vectors with 3 dim vector representation  of activity
# return[1] is only 3 dim vector representing activity. I haven't used it, but it may be useful later.
           
def gestures_array_to_vector(gestures_dir_array):
    #print(gestures_dir_array)
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


#trace_length_to_dictionary()
#trace_dir_array = find_all_files_in_range(20,54)
#vectors_array = gestures_array_to_vector(trace_dir_array)[0]
vectors_array = gestures_array_to_vector(uipackages)[0]

#print(vectors_array[0])


from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten, Dropout, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

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
K = 1

i = Input(shape = (T,D))
x = SimpleRNN(M, return_sequences=True)(i)
#x = GRU(M, return_sequences=True)(i)
#x = LSTM(M, return_sequences=True)(i)
x = GlobalMaxPool1D()(x)
x = Dropout(0.5)(x)
x = Dense(K,activation = 'relu')(x)
model = Model(i,x)
#model.compile( loss = 'mse', metrics = ['accuracy'], optimizer = Adam(lr = 0.001),)
model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr=0.001), metrics = ['accuracy'],)


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
            y_input = dataset[y_index,:][64]
            X.append(x_input)
            y.append(y_input)
            x_index +=1
            y_index +=1
    #return array(X), array(y)
    return np.asarray(X), np.asarray(y)



X, y = split_dataset_array(vectors_array, T)
#print(X.shape)
#print(y.shape)
#print(y)
r = model.fit(X, y, epochs = 200, validation_split = 0.4)


#import matplotlib.pyplot as plt
f1 = plt.figure(1)
plt.title('Loss')
plt.plot(r.history['loss'], label = 'train')
plt.plot(r.history['val_loss'], label = 'test')
plt.legend()
f1.show()

f2 = plt.figure(2)
plt.title('Accuracy')
#plt.plot(r.history['acc'], label = 'train')
#plt.plot(r.history['val_acc'], label = 'test')
plt.plot(r.history['accuracy'], label = 'train')
plt.plot(r.history['val_accuracy'], label = 'test')
plt.legend()
f2.show()


#prediction test

#test = ['filtered_traces/com.linkedin.android/trace_0/gestures.json']
packagename = uipackages[0]
test = []
test.append(packagename)
res = gestures_array_to_vector(test)
#testing for a ui
#The model just predict every next activity to 0
for i in range(len(res[0][0])):
    cor = []
    x = res[0][0][i].reshape(1,1,67)
    c = res[0][0][i]
    #cor.append(c[-2])
    #cor.append(c[-1])
    yhat = model.predict(x)
    #print(yhat)
    #print (yhat.argmax(axis=-1))