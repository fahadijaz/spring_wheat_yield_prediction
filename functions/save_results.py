import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime as dt

def save_results(model, agg_method, train_field, test_field,
                 features_all, importances, RMSE_test,RMSE_train,
                 R2_test, R2_train, GKF_CV):

    # features_all = [training_features,base_indices,spectral_indices_all,spectral_indices,weather_features,export_path]
    date_time = dt.now()
    train_feat = []
    if set(features_all[2]) <= set(features_all[0]): # spectral indices all
        train_feat.append('spectral_indices_all')
    elif set(features_all[3]) <= set(features_all[0]): # spectral indices
        train_feat.append('spectral_indices_select')
    if set(features_all[4]) <= set(features_all[0]): # weather features
        train_feat.append('weather_features')
    if set(features_all[1]) <= set(features_all[0]): # base indices
        train_feat.append('base_indices')
    if set(['Staur_Env', 'Vollebekk_Env']) <= set(features_all[0]): # environment variables
        train_feat.append('Environment_feature')
        
    results = {'Model': model,
               'Aggregation_method': agg_method,
               'Train_field': train_field,
               'Test_field': test_field,
               'Training_features': train_feat,
               'Feature_Importances': importances,
               'RMSE_test': RMSE_test,
               'RMSE_train': RMSE_train,
               'R2_test': R2_test,
               'R2_train': R2_train,
               'GKF_CV': GKF_CV,
               'DataTime': date_time}
    export_path = features_all[5]

    filename = export_path + 'results.csv'

    with open(filename, "a+") as csvfile:
        headers = results.keys()
        writer = csv.DictWriter(csvfile, delimiter=',',
                                lineterminator='\n', fieldnames=headers)

        # Check is the file is empty or not
        fileEmpty = os.stat(filename).st_size == 0
        # If empty, then add header
        if fileEmpty:
            writer.writeheader()  # file doesn't exist yet, write a header

        # Write the current data as next row
        writer.writerow(results)

def save_grid_results(list_zip, features_all=None):
    list_zip_list = list(list_zip[0])
    
    # Appending None entries to list to makeup for missing parameters
    while len(list_zip_list)<14:
        list_zip_list.append(None)
    
    model = list_zip_list[0]
    pipe = list_zip_list[1]
    train_score = list_zip_list[2]
    test_score = list_zip_list[3]
    p1 = list_zip_list[4]
    p2 = list_zip_list[5]
    p3 = list_zip_list[6]
    p4 = list_zip_list[7]
    p5 = list_zip_list[8]
    p6 = list_zip_list[9]
    p7 = list_zip_list[10]
    p8 = list_zip_list[11]
    p9 = list_zip_list[12]
    p10 = list_zip_list[13]

    
    # features_all = [training_features,base_indices,spectral_indices_all,spectral_indices,weather_features,export_path]
    date_time = dt.now()
    train_feat = []
    if set(features_all[2]) <= set(features_all[0]): # spectral indices all
        train_feat.append('spectral_indices_all')
    elif set(features_all[3]) <= set(features_all[0]): # spectral indices
        train_feat.append('spectral_indices_select')
    if set(features_all[4]) <= set(features_all[0]): # weather features
        train_feat.append('weather_features')
    if set(features_all[1]) <= set(features_all[0]): # base indices
        train_feat.append('base_indices')
    if set(['Staur_Env', 'Vollebekk_Env']) <= set(features_all[0]): # environment variables
        train_feat.append('Environment_feature')
        
        
    results = {'Model': model,
               'Pipeline': pipe,
               'Train_score': train_score,
               'Test_score': test_score,
               'Parameter_1': p1,
               'Parameter_2': p2,
               'Parameter_3': p3,
               'Parameter_4': p4,
               'Parameter_5': p5,
               'Parameter_6': p6,
               'Parameter_7': p7,
               'Parameter_8': p8,
               'Parameter_9': p9,
               'Parameter_10': p10,
               'Aggregation_method': agg_method,
               'Training_features': train_feat,
               'DataTime': date_time}

    export_path = features_all[5]

    filename = export_path + 'results_loop_org.csv'

    with open(filename, "a+") as csvfile:
        headers = results.keys()
        writer = csv.DictWriter(csvfile, delimiter=',',
                                lineterminator='\n', fieldnames=headers)

        # Check is the file is empty or not
        fileEmpty = os.stat(filename).st_size == 0
        # If empty, then add header
        if fileEmpty:
            writer.writeheader()  # file doesn't exist yet, write a header

        # Write the current data as next row
        writer.writerow(results)
    