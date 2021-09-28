
#==============================================================================
# Defining the function to vaiidate the model with the test data and 
# get the results from regression evaluation metrices in sklearn
#==============================================================================
pred = []
acc = []
import pprint as pprint
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def test_data_regression(model, features, X_train, X_train_std, y_train, X_test, X_test_std, y_test):
    pred = []
    accuracy = {}
    accuracy_std = {}

    #==============================================================================
    # Make predictions for test set
    #==============================================================================

    # Predict classes for samples in test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    success = False
    while not success:
        try:
            importances = model.feature_importances_
            success = True
        except:
            importances = None
            pass
    
    # Training and predictions on standardised data
    model.fit(X_train_std, y_train)
    y_pred_std = model.predict(X_test_std)
    
    success_std = False
    while not success_std:
        try:
            importances_std = model.feature_importances_
            success_std = True
        except:
            importances_std = None
            pass
        
    #==============================================================================
    # Compute performance
    #==============================================================================
    
    mse = mean_squared_error(y_test, y_pred, squared=True)
    mse_std = mean_squared_error(y_test, y_pred_std, squared=True)
    accuracy['MSE'] = mse
    accuracy_std['MSE'] = mse_std
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse_std = mean_squared_error(y_test, y_pred_std, squared=False)
    accuracy['RMSE'] = rmse
    accuracy_std['RMSE'] = rmse_std

    r2 = r2_score(y_test, y_pred)
    r2_std = r2_score(y_test, y_pred_std)
    accuracy['R2 Score'] = r2
    accuracy_std['R2 Score'] = r2_std

    acc.append(accuracy)
    # Print accuracy computed from predictions on the test set
    pp = pprint.PrettyPrinter(indent=4, width=80, depth=None, stream=None, compact=True, sort_dicts=False)
    pp.pprint(accuracy)
    
    return accuracy, accuracy_std, importances, importances_std 