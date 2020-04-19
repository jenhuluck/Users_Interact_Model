# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:59:31 2020

@author: CanXd
"""

import numpy as np
import json
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
"""

# Opening JSON file and loading the data 
# into the variable data 
with open('ui_names.json') as json_file: 
	data_names = json.load(json_file) 
ui_names = data_names['ui_names']
ui_names = np.array(ui_names)

#names = pd.DataFrame(ui_names, columns = ['name'])

#ui vectors
ui_vectors = np.load('ui_vectors.npy')

"""
#Tried kmean cluster


#choose ui
chosen_index = 3
chosen_ui = ui_vectors[chosen_index,:]
#print(ui_vectors[3,:])

# import KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=20)
clusters = kmeans.fit_predict(ui_vectors)
print(kmeans.cluster_centers_.shape)
cluFrame = pd.DataFrame(clusters, columns = ['cluster'])

#combine ui names and cluster results
cluFrame = pd.merge(names, cluFrame, on=None, left_on=None, right_on=None, left_index=True, right_index=True)
cluFrame['cluster'].value_counts()
print(cluFrame['cluster'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(20, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

"""
import pickle
import faiss

"""
#Create pickle data file
with open('ui.pickle', 'wb') as f:
    pickle.dump({"name": ui_names, "vector": ui_vectors}, f)
"""

def load_data():
    with open('ui.pickle', 'rb') as f:
        data_ui = pickle.load(f)
    return data_ui

data_ui = load_data()
#print(data_ui)

class ExactIndex():
    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.labels = labels    
   
    def build(self):
        self.index = faiss.IndexFlatL2(self.dimension,)
        self.index.add(self.vectors)
        
    def query(self, vectors, k=15):
    	#print(vectors)
    	v = np.array([vectors])
        distances, indices = self.index.search(v, k)
        # I expect only query on one vector thus the slice
        return [self.labels[i] for i in indices[0]]

index = ExactIndex(data_ui["vector"], data_ui["name"])
index.build()

print(index.query(data_ui['vector'][43183]))