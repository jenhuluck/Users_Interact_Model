# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:37:36 2020

@author: Jieyun Hu
"""

#This file is for trace statistical analysis
#put the file outside of the filtered_traces directory

import os
import json
from matplotlib import pyplot as plt
import numpy as np
import collections 
import pandas as pd

#print(os.getcwd())
# 9385 trace files
lst = os.listdir("filtered_traces") # current directory
def number_of_files():    
    number_files = len(lst)
    print(number_files)

#total trace number
def number_of_traces():
    total_trace = 0
    for f in lst:
        sublst = os.listdir("filtered_traces/"+f)
        total_trace = total_trace +len(sublst)  
    print(total_trace)

#save the number of trace to an array
# the index of the array is the total number of the trace
# the max traces is 53, caculated already
bucket = [0]*54
def traces_statistic():    
    #max = 0
    for f in lst:
        sublst = os.listdir("filtered_traces/"+f)
        for sub in sublst:
            file_name = "filtered_traces/"+f+"/"+sub+"/gestures.json"
            with open(file_name) as json_file:
                data = json.load(json_file)
                data_len = len(data)
                bucket[data_len]+=1
                #if data_len > max:
                    #max = data_len
                
    #print(bucket)     
#[0, 1740, 1373, 1116, 959, 800, 673, 528, 484, 358, 351, 285, 221, 215, 174, 163, 107, 119, 97, 83, 56, 46, 46, 45, 33, 19, 29, 27, 14, 29, 15, 13, 8, 9, 10, 4, 6, 8, 7, 4, 3, 1, 2, 1, 5, 1, 1, 2, 0, 0, 1, 0, 0, 1]                
def draw_bar(array, x_label, y_label, _title):
    plt.bar(np.arange(54),array)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(_title)
    plt.show()
    

#accumulate the numbers in the array from the end
def accumulate_array(array):
    high = len(array)
    sum = 0
    for i in reversed(range(high)):
        array[i]+= sum
        sum = array[i]        
    #print(array)

# save the file names with the same length of trace to dictionary
# key: length of the trace, obj : file names
    

dict = collections.defaultdict(list)

def trace_length_to_dictionary():
    for f in lst:
        sublst = os.listdir("filtered_traces/"+f)
        for sub in sublst:
            file_name = "filtered_traces/"+f+"/"+sub+"/gestures.json"
            with open(file_name) as json_file:
                data = json.load(json_file)
                data_len = len(data)
                dict[data_len].append(f)
                #dict[data_len].append(file_name)
    
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

# tp find the categories in app_details by query a list of apps
def panda_filter(array):
    app = pd.read_csv('app_details.csv') 
    app.columns = [column.replace(" ", "_") for column in app.columns ]    
    #app.query('App_Package_Name == "emotion.onekm"', inplace = True)
    
    filtered = app[app['App_Package_Name'].isin(array)]    
    return filtered['Category'].values.tolist()

def list_to_dict(array):
    dict_cat = collections.defaultdict(int)
    for c in array:
        dict_cat[c]+=1
    return dict_cat
        
def draw_bar_category(list_of_category):
    dict_cat = list_to_dict(list_of_category)   
    keys = dict_cat.keys()    
    vals = dict_cat.values()
    plt.xticks(rotation=90)
    plt.bar(keys, vals)
    
#number_of_traces()    
#traces_statistic()
#draw_bar(bucket, "length of trace", "number of trace",  "The distribution of trace length")
        
#accumulate_array(bucket)
#print(bucket)
#[10292, 10292, 8552, 7179, 6063, 5104, 4304, 3631, 3103, 2619, 2261, 1910, 1625, 1404, 1189, 1015, 852, 745, 626, 529, 446, 390, 344, 298, 253, 220, 201, 172, 145, 131, 102, 87, 74, 66, 57, 47, 43, 37, 29, 22, 18, 15, 14, 12, 11, 6, 5, 4, 2, 2, 2, 1, 1, 1]
#draw_bar(bucket, "accumulative length of trace", "number of trace",  "The distribution of accumulative trace length")

trace_length_to_dictionary()

list_of_trace = find_all_files_in_range(1,2)

filtered_array= panda_filter(list_of_trace)

draw_bar_category(filtered_array)


print(list_to_dict(filtered_array))



    
    
        
    

        