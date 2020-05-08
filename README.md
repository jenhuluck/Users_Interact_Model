# users_interact_model
The project used RICO datasets, where you can find:
https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/traces.tar.gz   
https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_layout_vectors.zip
https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/app_details.csv

Unzip the datasets First. 
Put the python files outside the folder filtered_traces and ui_layout_vectors, but under the same folder of app_details.csv,
then run each python file to see the result. 


encoding_modeling.py is to predict next activity (click or swipe). It includes data processing for general data, RNN modeling, evaluation and prediction.
encoding_modeling2.py is to predict the coordinates of next activity.
encoding_modeling_similarUI.py is to predict next activity, but use similar UI as input dataset.
ui_neighbors.py is to find similar UI using KNN.
trace_explore.py is for trace data analysis.
