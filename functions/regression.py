import numpy as np
import pandas as pd
from datetime import datetime as dt
# Pre.Processing
from sklearn.preprocessing import StandardScaler
# Pipeline
from sklearn.pipeline import make_pipeline
# Models
from sklearn.model_selection import GridSearchCV
# Metrices
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import os
import random
# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

def training_gkf_std(model, X, y, gkf=5):
    
    current_model = make_pipeline(StandardScaler(), model)

    scores = cross_validate(current_model, X, y, cv=gkf,
                            scoring=('r2', 'neg_root_mean_squared_error'),
                            return_train_score=True)
#     RMSE_test = "%0.2f (+/- %0.2f)" % (-1*scores['test_neg_root_mean_squared_error'].mean(), 
#                                   -1*scores['test_neg_root_mean_squared_error'].std() * 2)
#     RMSE_train = "%0.2f (+/- %0.2f)" % (-1*scores['train_neg_root_mean_squared_error'].mean(), 
#                                   -1*scores['train_neg_root_mean_squared_error'].std() * 2)


#     R2_test = "%0.2f (+/- %0.2f)" % (scores['test_r2'].mean(), 
#                                   scores['test_r2'].std() * 2)
#     R2_train = "%0.2f (+/- %0.2f)" % (scores['train_r2'].mean(), 
#                                   scores['train_r2'].std() * 2)

    RMSE_test = "%0.6f" % (-1*scores['test_neg_root_mean_squared_error'].mean())
    RMSE_train = "%0.6f" % (-1*scores['train_neg_root_mean_squared_error'].mean())


    R2_test = "%0.6f" % (scores['test_r2'].mean())
    R2_train = "%0.6f" % (scores['train_r2'].mean())
    
    print(str(model).split('()')[0])
    print(current_model)
    print(' RMSE Test:', RMSE_test, '       R2 Test:', R2_test)
    print('RMSE Train:', RMSE_train, '      R2 Train:', R2_train)
    
    # Feature importance
    current_model.fit(X, y)
    success = False
    while not success:
        try:
            feature_importance = current_model.steps[1][1].feature_importances_
            success = True
        except:
            feature_importance = None
            break

    # Saving results
    GKF_CV = True
    return feature_importance, RMSE_test, RMSE_train, R2_test, R2_train, GKF_CV
    
    
def training_regr(model, X_train, y_train, X_test, y_test, scatter_params):
    
    
    scatter_plot = scatter_params[0]
    scatter_title = scatter_params[1]
    scatter_save = scatter_params[2]
    export_path = scatter_params[3]
    feature_group = scatter_params[4]
    marker_size = scatter_params[5]
    fig_size = scatter_params[6]
    
    model_name =  str(model).split('(')[0]
    model_name_full = ''.join([ c if c.isalnum() else "_" for c in str(model).replace(" ", "") ])
    
    current_model = make_pipeline(StandardScaler(), model)

    current_model.fit(X_train, y_train)
    y_pred = current_model.predict(X_test)
    y_pred_train = current_model.predict(X_train)

    RMSE_test = "%0.6f" % (mean_squared_error(y_test, y_pred, squared=False))
    RMSE_test_short = "%0.3f" % (mean_squared_error(y_test, y_pred, squared=False))
    RMSE_train = "%0.6f" % (mean_squared_error(y_train, y_pred_train, squared=False))


    R2_test = "%0.6f" % (r2_score(y_test, y_pred))
    R2_test_short = "%0.3f" % (r2_score(y_test, y_pred))
    R2_train = "%0.6f" % (r2_score(y_train, y_pred_train))
    
    print(str(model).split('()')[0])
    print(current_model)
    print(' RMSE Test:', RMSE_test, '       R2 Test:', R2_test)
    print('RMSE Train:', RMSE_train, '      R2 Train:', R2_train)
    
    # Saving plots
    export_plots = export_path+ '/Scatter_Plots/' +'/Feat_Grp_'+str(feature_group)+'/'+model_name_full+'/'
    os.makedirs(export_plots, exist_ok=True)
    
    # Scatter plot of results
    if scatter_plot:

        fig, ax = plt.subplots(figsize=fig_size)
        
        ax.scatter(y_pred, y_test, s=marker_size)
        
        # plotting straight line
        max_range = int(max(max(y_pred), max(y_test)))
        min_range = int(min(min(y_pred), min(y_test)))
        
        x = np.linspace(min_range, max_range, 50)
        ax.plot(x, x, 'r')
        
        ax.set_title(model_name + ' ' + scatter_title,
                    fontdict = {'fontsize' : 11})
        ax.set_xlabel('Predicted grain yield (kg/daa)')
        ax.set_ylabel('Measured grain yield (kg/daa)')
        text = model_name + '\n' + 'R\u00b2 = '+ R2_test_short + '\n' + 'RMSE = ' + RMSE_test_short
        ax.text(.01, .98, text, ha='left', va='top', weight=100, color='black', fontsize ='large', transform=ax.transAxes)

        if scatter_save:
            # There was peoblem with saving plots in a loop where 66-33 ratio was used for same df
            # Probably beacuse of to long names/path address
            # So, using try method to avoid errors and saving plots with slightly different names
            filename = '/Scatter_'+str(scatter_title)+'-feat_grp_'+str(feature_group)+'.jpg'
            alt_filename = '/Sct'+str(scatter_title)+'g'+str(feature_group)+'.jpg'
            alt_filename_2 = '/'+str(random.randint(100, 999))
            success = False
            while not success:
                try:
                    plt.savefig(export_plots+filename, dpi=150, bbox_inches='tight')
                    success = True
                except:
                    try:
                        plt.savefig(export_plots+alt_filename, dpi=150, bbox_inches='tight')
                        success = True
                    except:
                        plt.savefig(export_plots+alt_filename_2, dpi=150, bbox_inches='tight')
                        break
        plt.show()
    
    # Feature importance
    success = False
    while not success:
        try:
            feature_importance = current_model.steps[1][1].feature_importances_
            success = True
        except:
            feature_importance = None
            break

    GKF_CV = False
    
    return feature_importance, RMSE_test, RMSE_train, R2_test, R2_train, GKF_CV, y_pred


def grid(Xtrain,
         ytrain,
         estimator,
         params_grid,
         scores,
         cvs,
         cores,
         verb):

    t1 = time.time()

    gs = GridSearchCV(estimator=estimator,
                      param_grid=params_grid,
                      scoring=scores,
                      cv=cvs,
                      n_jobs=cores,
                      verbose=verb,
                     return_train_score=True)

    gs = gs.fit(Xtrain, ytrain)
    print(estimator)
    print(gs.best_score_)
    print(gs.best_params_)
    
    t2 = time.time()

    # Saving results to csv file
    results = []
    import datetime
    datetime = dt.now()

    results.append((np.array((gs.best_estimator_, gs, score, gs.best_score_, gs.best_params_, 
                              gs.cv_results_['mean_train_score'].mean(),
                             ((t2 - t1) / 60), datetime), dtype=object)))

    pd.DataFrame(np.asarray(results)).to_csv(export_path+'results_grid.csv',
                                             mode='a',
                                             header=None)

    print('Total time: ', (t2 - t1) / 60, 'minutes')