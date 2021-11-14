import numpy as np
import pandas as pd
from datetime import datetime as dt

def list_test_train_df(all_df_, train_field, test_field, year):
    # Returns a string list of train dfs and test dfs. Not conct
    # Need to be conct afterwards
    
    # year = '2019', '2020', 'all' str
#     train_field = 'Vollebekk' , 'Staur'
#     test_field = 'Vollebekk' , 'Staur'

    # Asserting if the user has given the right inputs
    assert train_field != test_field
    assert train_field == 'Vollebekk' or train_field == 'Staur'
    assert test_field == 'Vollebekk' or test_field == 'Staur'
    assert year == '2019' or year == '2020' or year == 'all'

    # Filtering based on year
    all_df_temp1 = [x for x in all_df_ if not 'Robot' in x]
    if not year == 'all':
        all_df_temp = [x for x in all_df_temp1 if year in x]
    else:
        all_df_temp = all_df_temp1.copy()
        
    # Making list of training dfs for conct before training
    staur_list = []
    for x in all_df_temp:
        if 'Staur' in x:
            staur_list.append(x)

    # Making list of test dfs for conct before training
    vollebekk_list = []
    for x in all_df_temp:
        if not 'Staur' in x and not 'Robot' in x:
            vollebekk_list.append(x)
    
    train_str_list = []
    test_str_list = []
    # Assigning test and train sets based on given inputs
    if train_field == 'Staur':
        train_str_list = staur_list.copy()
        print('Training data:', staur_list)
        
        test_str_list = vollebekk_list.copy()
        print('Test data:', vollebekk_list)
    elif train_field == 'Vollebekk':
        train_str_list = vollebekk_list.copy()
        print('Training data:', vollebekk_list)
        
        test_str_list = staur_list.copy()
        print('Test data:', staur_list)
    else:
        raise NameError
    
    return (train_str_list, test_str_list)


# data_prep_field(all_df_, train_field = ['Staur', 'Masbasis'], test_field = ['Staur', 'Masbasis'], 
#                 year_train = ['2019', 2020], year_test = ['2019', 2020]):

def data_prep_field(all_df_, train_field, test_field, year):
    
    # year = '2019', '2020', 'all' str
#     train_field = 'Vollebekk' , 'Staur'
#     test_field = 'Vollebekk' , 'Staur'

    # Asserting if the user has given the right inputs
    assert train_field != test_field
    assert train_field == 'Vollebekk' or train_field == 'Staur'
    assert test_field == 'Vollebekk' or test_field == 'Staur'
    assert year == '2019' or year == '2020' or year == 'all'

    # Filtering based on year
    all_df_temp1 = [x for x in all_df_ if not 'Robot' in x]
    if not year == 'all':
        all_df_temp = [x for x in all_df_temp1 if year in x]
    else:
        all_df_temp = all_df_temp1.copy()
        
    # Making list of training dfs for conct before training
    staur_df_list = []
    staur_list = []
    print(all_df_temp)
    for x in all_df_temp:
        if 'Staur' in x:
            staur_list.append(x)
            print(staur_list)
#             staur_df_list.append(locals()[x])

    # Making list of test dfs for conct before training
    vollebekk_df_list = []
    vollebekk_list = []
    for x in all_df_temp:
        if not 'Staur' in x and not 'Robot' in x:
            vollebekk_list.append(x)
#             vollebekk_df_list.append(locals()[x])
    
    # Assigning test and train sets based on given inputs
    if train_field == 'Staur':
        train_df_list = staur_df_list.copy()
        print('Training data:', staur_list)
        
        test_df_list = vollebekk_df_list.copy()
        print('Test data:', vollebekk_list)
    elif train_field == 'Vollebekk':
        train_df_list = vollebekk_df_list.copy()
        print('Training data:', vollebekk_list)
        
        test_df_list = staur_df_list.copy()
        print('Test data:', staur_list)
    else:
        raise NameError
        
    train_df = pd.concat(train_df_list)
    test_df = pd.concat(test_df_list)

    X_train = train_df[training_features]
    y_train = train_df[target_features].values.ravel()
    X_test = test_df[training_features]
    y_test = test_df[target_features].values.ravel()
    
    return X_train, y_train, X_test, y_test

    