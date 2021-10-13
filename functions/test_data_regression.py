from functions.plot_feature_importance import plot_feature_importance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import math
import datetime
import numpy as np
import pandas as pd


# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#==============================================================================
# Defining the function to vaiidate the model with the test data and 
# get the results from regression evaluation metrices in sklearn
#==============================================================================
pred = []
acc = []

    

def test_train_split(X, y, split_, rs = 100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_, random_state=rs)
    return X_train, X_test, y_train, y_test

def std_scaler(X_train, X_test):

def train_pred_model(model,  X_train, y_train, X_test, y_test):
    temp_model = model
    temp_model.fit(X_train, y_train)
    y_pred = temp_model.predict(X_test)
    y_pred_train = temp_model.predict(X_train)
    
    return temp_model, y_pred, y_pred_train


def feature_imp(model):
success = False
while not success:
    try:
        importances = model.feature_importances_
        success = True
        plot_feature_importance(importance,features,model)
    except:
        importances = None
        pass
    
def rmse_():

def r2_score_():


def plot_feat_imp(importance, features, model):


def test_data_regression():
    pred = []
    accuracy = {}
    accuracy_std = {}
    accuracy_train = {}
    #==============================================================================
    # Make predictions for test set
    #==============================================================================

    # Predict classes for samples in test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    success = False
    while not success:
        try:
            importances = model.feature_importances_
            success = True
            plot_feature_importance(importance,features,model)
        except:
            importances = None
            pass
    
    # Training and predictions on standardised data
    # model.fit(X_train_std, y_train)
    # y_pred_std = model.predict(X_test_std)
    
    # success_std = False
    # while not success_std:
     #    try:
      #       importances_std = model.feature_importances_
       #      success_std = True
            
    #     except:
    #         importances_std = None
    #         pass
        
    #==============================================================================
    # Compute performance
    #==============================================================================
    
    mse = mean_squared_error(y_test, y_pred, squared=True)
    #mse_std = mean_squared_error(y_test, y_pred_std, squared=True)
    mse_train = mean_squared_error(y_train, y_pred_train, squared=True)

    accuracy['MSE'] = mse
    #accuracy_std['MSE'] = mse_std
    accuracy_train['MSE'] = mse_train

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    #rmse_std = mean_squared_error(y_test, y_pred_std, squared=False)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)

    accuracy['RMSE'] = rmse
    #accuracy_std['RMSE'] = rmse_std
    accuracy_train['RMSE'] = rmse_train

    r2 = r2_score(y_test, y_pred)
    #r2_std = r2_score(y_test, y_pred_std)
    r2_train = r2_score(y_train, y_pred_train)

    accuracy['R2 Score'] = r2
    #accuracy_std['R2 Score'] = r2_std
    accuracy_train['R2 Score'] = r2_train

    # acc.append(accuracy)
    # Print accuracy computed from predictions on the test set
    pp = pprint.PrettyPrinter(indent=4, width=80, depth=None, stream=None, compact=True, sort_dicts=False)
    pp.pprint(accuracy)
    pp.pprint(accuracy_train)

    return accuracy, accuracy_train, importances