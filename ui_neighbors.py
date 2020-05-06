# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:14:15 2020

@author: Xueyuan Chen
"""

"""
samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(samples)

print(neigh.kneighbors([[1., 1., 1.]]))

X = [[1., 1., 1.]]
print(neigh.kneighbors(X, return_distance=False))
"""
import os
import numpy as np
import json
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
    
    print(retrieved_ui)
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
    
    print(uipackages)
    return uipackages

import shutil, errno

def copy_retrievedUIs_toDst(src, dst):
    filelist = [ f for f in os.listdir(dst) if f.endswith(".jpg") ]
    for f in filelist:
        os.remove(os.path.join(dst, f))
    
    for ui in ui_retrieved:
        uinum = int(re.search(r'\d+', ui).group())
        package = ui_details[ui_details['ui_num']==uinum]['package_name'].values
        if package:
            shutil.copy(src+str(uinum)+'.jpg', dst)


#ui_query = get_query_ui_vector('36043.png') #lists
#ui_query = get_query_ui_vector('36171.png') #lists with larger graph
#ui_query = get_query_ui_vector('36048.png') #login screen
#ui_query = get_query_ui_vector('11374.png') #calendar
ui_query = get_query_ui_vector('21506.png') #image grids
ui_retrieved = find_similar_ui_names(ui_query, 50)
#uipackages = get_package_names(ui_retrieved)

src_path = 'combined/'
dst_path = 'retrieved_UIs'
copy_retrievedUIs_toDst(src_path, dst_path)