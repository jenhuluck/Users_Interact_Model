# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:17:04 2020

@author: CanXd
"""

import numpy as np
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# Opening JSON file and loading the data 
# into the variable data 
with open('ui_names.json') as json_file: 
	data_names = json.load(json_file) 

#ui names dataframe
names = data_names['ui_names']
names = np.array(names)
ui_names = pd.DataFrame(names, columns = ['name'])

#ui details dataframe
ui_details = pd.read_csv('ui_details.csv')
ui_details.columns = ['ui_num','package_name','interaction_trace','ui_trace_num']
num = ui_details[ui_details['ui_trace_num']==29]['ui_num'].values
#print(num)
index = ui_details.index[ui_details['package_name']=='com.bamnetworks.mobile.android.ballpark']
#print(index)
#print(index[0])
s = ui_details['ui_num']
#print(s[index[0]])

#get package name


#get index of file
idx = ui_names.index[ui_names['name']=='11374.png']
print(idx[0])
