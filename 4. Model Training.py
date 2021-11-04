import sys
# prints the location of current python env
print(sys.prefix)
# print which python executable am I running
sys.executable

# In Anaconda Prompt, run the following
# where python
# where pip

# to find out where are the python executables located on disk


# import sklearn
# print (sklearn.__version__)
# !pip list -o

import os
import csv
from datetime import datetime as dt
import numpy as np
import pandas as pd

# Dictionaries
import json
from pprint import pprint

# Iterate in loops
import itertools
from itertools import zip_longest


# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Display rows and columns Pandas
pd.options.display.max_columns = 100
pd.set_option('display.max_rows',100)

# # For displaying max rows in series
# pd.options.display.max_rows = 10

import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance

# Prints the current working directory
os.getcwd()
# os.listdir()

username = str(os.getcwd()).split('\\')[2]
user_path = r'C:/Users/'+username+'/'
username, user_path

main_path = r'C:\Users\fahad\MegaSync\NMBU\GitHub\vPheno/Data/'
path = r'C:\Users\fahad\MegaSync\NMBU\GitHub\vPheno//Data/3. merged data/'
export_path = r'C:\Users\fahad\MegaSync\NMBU\GitHub\vPheno//Data/4. results/'
export_path_comparability = r'C:\Users\fahad\MegaSync\NMBU\GitHub\vPheno//Data/4. results/comparability/'


# Create export_path folder if not exists already
os.makedirs(path, exist_ok=True)
os.makedirs(export_path, exist_ok=True)
os.makedirs(export_path_comparability, exist_ok=True)

os.listdir(path)

# Making dictionary of files in each folder, in case there are multiple types of data
dict_paths = {}
def explore(starting_path):
    for dirpath, dirnames, filenames in os.walk(starting_path):
        dict_paths[dirpath.split('/')[-2]] = filenames
#     pprint(dict_paths)
explore(path)

# Get the list of all files in directory tree at given path

files_with_address = []
files_list = []

for (dirpath, dirnames, filenames) in os.walk(path):
    files_with_address += [os.path.join(dirpath, file) for file in filenames]
    files_list.extend(filenames)
    
print(len(files_with_address), 'files found in the directory')
# files_with_address
files_list

print('Total number of files are :', len(files_list))

print('Number of unique file names are:', len(set(files_list)))

print('There is/are', len(files_list) - len(set(files_list)),'duplicate file name/names.')
if len(files_list) - len(set(files_list)) > 0:
    raise NameError


all_df = []
for data in files_with_address:
    file_name = os.path.splitext(os.path.basename(data))[0]

    # Replce all invalid characters in the name
    file_name = file_name.replace(" ", "_")
    file_name = file_name.replace("-", "_")
    file_name = file_name.replace(")", "")
    file_name = file_name.replace("(", "")
    df_name = file_name.replace(".", "")
    # Test: Check if the same date is already present in the current dict key
    if df_name in all_df:
        print(f'A file with the same name {df_name} has already been imported. \n Please check if there is duplication of data.')
        raise NameError
    all_df.append(df_name)

    locals()[df_name] = pd.read_csv(data, index_col=False)
    print(df_name, '=====', locals()[df_name].shape)
# all_df

all_df_simps = [x for x in all_df if 'Simps' in x]
all_df_trapz = [x for x in all_df if 'Trapz' in x]
all_df_simps, all_df_trapz

print(f'Total imported {len(all_df)}')
all_df

a_file = open(main_path+'vollebekk_weather_columns.json', "r")
output_str = a_file.read()
# The file is imported as string

# Converting it to python format
weather_cols_vollebekk = json.loads(output_str)
a_file.close()

pprint(len(weather_cols_vollebekk))

a_file = open(main_path+'staur_weather_columns.json', "r")
output_str = a_file.read()
# The file is imported as string

# Converting it to python format
weather_cols_staur = json.loads(output_str)
a_file.close()

pprint(len(weather_cols_staur))

a_file = open(main_path+"yield_columns.json", "r")
output_str = a_file.read()

# The file is imported as string
# Converting it to python format
yield_cols = json.loads(output_str)
a_file.close()
print(yield_cols)

a_file = open(main_path+"spectral_indices_columns.json", "r")
output_str = a_file.read()

# The file is imported as string
# Converting it to python format
spectral_indices_all = json.loads(output_str)
a_file.close()
print(spectral_indices_all)

a_file = open(main_path+"base_indices_columns.json", "r")
output_str = a_file.read()

# The file is imported as string
# Converting it to python format
base_indices = json.loads(output_str)
a_file.close()
print(base_indices)

# ToDo: Add check for duplicate columns in the df

base_indices

spectral_indices_all 

drop_indices = ['EVI', 'GLI', 'MTCI']

spectral_indices = [x for x in spectral_indices_all if x not in drop_indices]

# Staur weather columns are all also present in Vollebekk weather so they can be use as general weather features
weather_features = weather_cols_staur.copy()

environment_var = weather_features + ['Staur_Env', 'Vollebekk_Env']

yield_cols




# # Dict for saving the name and location of the yield column/s
# loc_yield_cols = {}
# for df in all_df:
#     loc = 0
#     for cols in locals()[df].columns.tolist():
#         for y_col in yield_cols:
#             if not cols.find(y_col):
#                 loc_yield_cols[cols+'_'+df] = loc
#                 print(f'\"{cols}\" column in {df} is the yield column\n as it contains the text \"{y_col}\". It is located at location {loc}')
#         loc += 1

#     yield_cols_found = list(loc_yield_cols.keys())
#     target_cols=yield_cols_found[0]
# loc_yield_cols

# Dropping unnecessary columns

for df in all_df:
    temp_df = locals()[df].copy()
    locals()[df] = temp_df[base_indices+spectral_indices_all+environment_var+['Name','GrainYield']]
    print(df, temp_df.shape, '==>', locals()[df].shape)


# Dropping rows with missing value in any column

for df in all_df:
    temp_df = locals()[df].copy()
    locals()[df] = temp_df.dropna(axis=0)
    print(temp_df.shape[0] - locals()[df].shape[0], ' rows dropped in ', df)
#     print(locals()[df].shape[0])

# for col in base_indices+spectral_indices:
# #     col='Blue'
#     fig_size=(8, 5)
#     fig, ax = plt.subplots(figsize=fig_size)
#     plots = ax

#     for df in all_df_simps:
# #         if not 'Robot' in df and  not 'Staur' in df:
# #         if 'Gram' in df and  'Masb' in df:
# #             if '2020' in df:
#         temp_df = locals()[df].copy()
#         ax.boxplot(sorted(temp_df[col].values), positions = [all_df_simps.index(df)], labels=[df.split('_')[0][:5]+'_'+df.split('_')[1]])
# #         ax.plot(sorted(temp_df[col].values), label=df.split('_')[0]+'_'+df.split('_')[1])
#     # Printing the band/index name in plot of the fiels_sample for reference
#     text = col
#     ax.text(.95, .98, text, ha='center', va='top', weight=100, color='blue', fontsize ='xx-large', transform=ax.transAxes)

#     ax.legend(loc=1)
#     plt.tight_layout()
# #     plt.savefig(export_path_comparability+col+'_box.jpg',dpi=250, bbox_inches='tight', transform=ax.transAxes)
#     plt.show()
# #     break

# for col in base_indices+spectral_indices:
# #     col='Blue'
#     fig_size=(8, 5)
#     fig, ax = plt.subplots(figsize=fig_size)
#     plots = ax

#     for df in all_df_simps:
# #         if not 'Robot' in df and  not 'Staur' in df:
# #         if 'Gram' in df and  'Masb' in df:
# #             if '2020' in df:
#         temp_df = locals()[df].copy()
# #         ax.boxplot(sorted(temp_df[col].values), positions = [all_df_simps.index(df)], labels=[df.split('_')[0][:5]+'_'+df.split('_')[1]])
#         ax.plot(sorted(temp_df[col].values), label=df.split('_')[0]+'_'+df.split('_')[1])
#     # Printing the band/index name in plot of the fiels_sample for reference
#     text = col
#     ax.text(.87, .6, text, ha='center', va='top', weight=100, color='blue', fontsize ='xx-large', transform=ax.transAxes)

#     ax.legend(loc=1)
#     plt.tight_layout()
# #     plt.savefig(export_path_comparability+col+'_sorted.jpg',dpi=250, bbox_inches='tight', transform=ax.transAxes)
#     plt.show()
# #     break

# for col in base_indices+spectral_indices:
# #     col='Blue'
#     fig_size=(8, 5)
#     fig, ax = plt.subplots(figsize=fig_size)
#     plots = ax

#     for df in all_df_simps:
# #         if not 'Robot' in df and  not 'Staur' in df:
# #         if 'Gram' in df and  'Masb' in df:
# #             if '2020' in df:
#         temp_df = locals()[df].copy()
# #         ax.boxplot(sorted(temp_df[col].values), positions = [all_df_simps.index(df)], labels=[df.split('_')[0][:5]+'_'+df.split('_')[1]])
#         ax.plot((temp_df[col].values), label=df.split('_')[0]+'_'+df.split('_')[1])
#     # Printing the band/index name in plot of the fiels_sample for reference
#     text = col
#     ax.text(.87, .6, text, ha='center', va='top', weight=100, color='blue', fontsize ='xx-large', transform=ax.transAxes)

#     ax.legend(loc=1)
#     plt.tight_layout()
# #     plt.savefig(export_path_comparability+col+'_random.jpg',dpi=250, bbox_inches='tight', transform=ax.transAxes)
#     plt.show()
# #     break

from scipy.stats import zscore

for df in all_df:
    temp_df = locals()[df].copy()
    for col in temp_df.columns:
        # Checking if the column is not a yield column
        if col not in yield_cols+environment_var:
            temp_df[col] = zscore(temp_df[col])
    locals()[df] = temp_df.copy()
    print(df)

# for col in base_indices+spectral_indices:
# #     col='Blue'
#     fig_size=(8, 5)
#     fig, ax = plt.subplots(figsize=fig_size)
#     plots = ax

#     for df in all_df_simps:
# #         if not 'Robot' in df and  not 'Staur' in df:
# #         if 'Gram' in df and  'Masb' in df:
# #             if '2020' in df:
#         temp_df = locals()[df].copy()
#         ax.boxplot(sorted(temp_df[col].values), positions = [all_df_simps.index(df)], labels=[df.split('_')[0][:5]+'_'+df.split('_')[1]])
# #         ax.plot(sorted(temp_df[col].values), label=df.split('_')[0]+'_'+df.split('_')[1])
#     # Printing the band/index name in plot of the fiels_sample for reference
#     text = col
#     ax.text(.87, .6, text, ha='center', va='top', weight=100, color='blue', fontsize ='xx-large', transform=ax.transAxes)

#     ax.legend(loc=1)
#     plt.tight_layout()
# #     plt.savefig(export_path_comparability+col+'_box.jpg',dpi=250, bbox_inches='tight', transform=ax.transAxes)
#     plt.show()
# #     break

# for df in all_df_simps:
#     temp_df = locals()[df][base_indices+spectral_indices+['GrainYield']].copy()
#     data = temp_df.copy()
#     for col in base_indices:
#         print(df)
#         df_a = temp_df[col]
#         df_b = temp_df['GrainYield']


#         fig, ax = plt.subplots(1, figsize=(12,8))
#         sns.kdeplot(df_a, y=df_b, cmap='Blues',
#                    shade=True, thresh=0.05, clip=(-1,300))
#         plt.scatter(df_a, df_b, color='orangered')
#         plt.show()

# for df in all_df_simps:
#     print(df)
#     temp_df = locals()[df][['GrainYield']+spectral_indices].copy()
# #     temp_df = locals()[df][spectral_indices+['GrainYield']].copy()
#     data = temp_df
#     columns = temp_df.columns
#     corr = data.corr()
#     fig_size=(15,8)

#     fig, ax = plt.subplots(figsize=fig_size)
    
#     mask = np.triu(np.ones_like(corr, dtype=np.bool))

    
#     ax = sns.heatmap(
#         corr, mask=mask,
#         vmin=-1, vmax=1, center=0,
#         cmap=sns.diverging_palette(20, 220, n=200),
#         square=True
#     )    
    
#     ax.set_xticklabels(
#         ax.get_xticklabels(),
#         rotation=45,
#         horizontalalignment='right'
#     );
#     plt.show()





from datetime import datetime as dt


def save_results(model, agg_method, train_field, test_field,
                 training_features, importances, RMSE_test,
                 RMSE_train, R2_test, R2_train, GKF_CV):

    date_time = dt.now()
    train_feat = []
    if set(spectral_indices_all) <= set(training_features):
        train_feat.append('spectral_indices_all')
    elif set(spectral_indices) <= set(training_features):
        train_feat.append('spectral_indices_select')
    if set(weather_features) <= set(training_features):
        train_feat.append('weather_features')
    if set(base_indices) <= set(training_features):
        train_feat.append('base_indices')
    if set(['Staur_Env', 'Vollebekk_Env']) <= set(training_features):
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

    filename = export_path + 'results_org.csv'

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
    del(results, date_time)

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
    staur_df_list = []
    staur_list = []
    for x in all_df_temp:
        if 'Staur' in x:
            staur_list.append(x)
#             staur_df_list.append(locals()[x])

    # Making list of test dfs for conct before training
    vollebekk_df_list = []
    vollebekk_list = []
    for x in all_df_temp:
        if not 'Staur' in x and not 'Robot' in x:
            vollebekk_list.append(x)
#             vollebekk_df_list.append(locals()[x])
    
    train_df_list = []
    train_str_list = []
    test_df_list = []
    test_str_list = []
    # Assigning test and train sets based on given inputs
    if train_field == 'Staur':
#         train_df_list = staur_df_list.copy()
        train_str_list = staur_list.copy()
        print('Training data:', staur_list)
        
#         test_df_list = vollebekk_df_list.copy()
        test_str_list = vollebekk_list.copy()
        print('Test data:', vollebekk_list)
    elif train_field == 'Vollebekk':
#         train_df_list = vollebekk_df_list.copy()
        train_str_list = vollebekk_list.copy()
        print('Training data:', vollebekk_list)
        
#         test_df_list = staur_df_list.copy()
        test_str_list = staur_list.copy()
        print('Test data:', staur_list)
    else:
        raise NameError
    
    return (train_str_list, test_str_list)
    del (all_df_temp1, all_df_temp, staur_df_list, staur_list, vollebekk_df_list, vollebekk_list, train_df_list, train_str_list, test_df_list, test_str_list)

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

def training_gkf_std(model, X, y, gkf):
    
    current_model = make_pipeline(StandardScaler(), model)
#     current_model = make_pipeline(model)

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

    RMSE_test = "%0.2f" % (-1*scores['test_neg_root_mean_squared_error'].mean())
    RMSE_train = "%0.2f" % (-1*scores['train_neg_root_mean_squared_error'].mean())


    R2_test = "%0.2f" % (scores['test_r2'].mean())
    R2_train = "%0.2f" % (scores['train_r2'].mean())
    
    print(model)
    print(current_model)
    print(' RMSE Test:', RMSE_test, '       R2 Test:', R2_test)
    print('RMSE Train:', RMSE_train, '      R2 Train:', R2_train)
    
#     # Feature importance
#     current_model.fit(X, y)
#     success = False
#     while not success:
#         try:
#             feature_importance = current_model.steps[1][1].feature_importances_
#             success = True
#         except:
#             feature_importance = None
#             pass
    
    feature_importance = None

    # Saving results
    GKF_CV = gkf
    return feature_importance, RMSE_test, RMSE_train, R2_test, R2_train, GKF_CV

def training_regr(model, X_train, y_train, X_test, y_test):
    current_model = make_pipeline(StandardScaler(), model)
#     current_model = make_pipeline(model)

    current_model.fit(X_train, y_train)
    y_pred_train = current_model.predict(X_train)
    y_pred = current_model.predict(X_test)
    
    RMSE_test = mean_squared_error(y_test, y_pred, squared=False)
    RMSE_train = mean_squared_error(y_train, y_pred_train, squared=False)


    R2_test = r2_score(y_test, y_pred)
    R2_train = r2_score(y_train, y_pred_train)
    
    print(model)
    print(current_model)
    print(' RMSE Test:', RMSE_test, '       R2 Test:', R2_test)
    print('RMSE Train:', RMSE_train, '      R2 Train:', R2_train)

#     # Feature importance
#     success = False
#     while not success:
#         try:
#             feature_importance = current_model.steps[1][1].feature_importances_
#             success = True
#         except:
#             feature_importance = None
#             pass

    feature_importance = None

    GKF_CV = False
    
    return feature_importance, RMSE_test, RMSE_train, R2_test, R2_train, GKF_CV



# from matplotlib.backends.backend_pdf import PdfPages

# # Create plots folder if not exists already
# os.makedirs(plots_export_path, exist_ok=True)

# pdf = PdfPages(plots_export_path+'feat_imp.pdf')

def plot_feat_imp(feature_importance, model, train_feat, threshold='all', sort_feat=True):
    # threshold =  percentage of max(features_importance) or 'all' or top_x number of features
    # Plotting feature importance
    # Create arrays from feature importance and feature names

    feature_names = train_feat.copy()
    model_name =  str(model).split('(')[0]
    
    # Default threshold is 0, i.e. use all features
    thres = 0

    # Selecting features based on given threshold
    if isinstance(threshold, int) or isinstance(threshold, float):
        thres = threshold * 0.01
    elif str.lower(threshold) == 'all':
        thres = 0

    importances, names = zip(*(
        (x, y) for x, y in zip(feature_importance, feature_names) if x >= thres*max(feature_importance)))
    
    # Finding and filtering top_x number of features
    if isinstance(threshold, str):
        if str.lower(threshold.split('_')[0]) == 'top':
            top_x_feat = int(threshold.split('_')[1])
            sort_imp, sorted_name = zip(*sorted(zip(feature_importance, feature_names), reverse=True))

            importances, names = zip(*(
                (x, y) for x, y in zip(feature_importance, feature_names) if y in sorted_name[:top_x_feat]))  
    
    # Sorting faeture importances if required
    if sort_feat:
        importances, names = zip(*sorted(zip(importances, names), reverse=True))

    # Create a DataFrame using a Dictionary
    data={'feature_names':names,'feature_importance':importances}
    feat_imp_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
#     feat_imp_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,5))
    #Plot Searborn bar chart
    sns.barplot(y=feat_imp_df['feature_importance'], x=feat_imp_df['feature_names'], palette = 'winter'  )
    #Add chart labels

    plt.title(model_name + ' Feature Importance')
    plt.xticks(rotation=60)
    plt.xlabel('Feature Names')
    plt.ylabel('Feature Importance')
    export_plots = export_path+'/Feature_Importance/'
    os.makedirs(export_plots, exist_ok=True)
#     plt.savefig(export_plots+'feature_importance'+model_name+'.jpg',dpi=150, bbox_inches='tight')
#     plt.savefig(export_plots+col+feature_importance_'+model_name+'.pdf',dpi=500, bbox_inches='tight')

    plt.show()
    
    

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
                      refit=scores[1],
                     return_train_score=True)

    gs = gs.fit(Xtrain, ytrain)
    print(estimator)
    print(gs.best_score_)
    print(gs.best_params_)
    
    t2 = time.time()

    # Saving results to csv file
    results = []
    import datetime
    datetime = datetime.datetime.now()

    results.append((np.array((gs.best_estimator_, gs, score, gs.best_score_, gs.best_params_, 
                              gs.cv_results_['mean_train_score'].mean(), param_grid, cvs, 
                             ((t2 - t1) / 60), datetime), dtype=object)))

    pd.DataFrame(np.asarray(results)).to_csv(export_path+'results_grid.csv',
                                             mode='a',
                                             header=None)

    print('Total time: ', (t2 - t1) / 60, 'minutes')

# Number of cores in the system being used
import multiprocessing
multiprocessing.cpu_count()

import psutil
psutil.cpu_count()

import cpuinfo
info = cpuinfo.get_cpu_info()
print('python_version:', info['python_version'])
print(info['arch'])
print(info['bits'])
print(info['count'])
print(info['arch_string_raw'])
print(info['vendor_id_raw'])
print(info['brand_raw'])
print(info['hz_advertised_friendly'])




from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time
from datetime import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
# import xgboost as xgb
from sklearn.linear_model import Lasso
# from catboost import CatBoostRegressor


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


threshold_all = 'top_25'
sorted_all = True
agg_method = 'Simpsons'
# agg_method = 'Trapezoid'

# training_features = base_indices + spectral_indices + environment_var
# training_features = base_indices + spectral_indices + weather_features
training_features =  spectral_indices + weather_features
# training_features = spectral_indices

target_features = ['GrainYield']

group_feature = ['Name']

if agg_method == 'Simpsons':
    all_df_now = all_df_simps.copy()
elif agg_method == 'Trapezoid': 
    all_df_now = all_df_trapz.copy()

scores = ['neg_root_mean_squared_error', 'r2']
cv = 3
core = 6
verbos = 8

temp_list = [x for x in all_df_simps if not 'Robot' in x]

# Making list of df for conct before training
# This is different form list of srtings, as this is a list of actual dataframes
df_list = []
for x in temp_list:
    df_list.append(locals()[x])

df_ = pd.concat(df_list)

X = df_[training_features]
y = df_[target_features].values.ravel()
groups = df_[group_feature].values.ravel()

gkf = list(GroupKFold(n_splits=6).split(X, y, groups))
# gkf = list(StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=1).split(X, y, groups))


#==============================================================================
# RandomForestRegressor
#==============================================================================
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__max_depth' : [x for x in range(1, 10)],
                  'model__max_features' : ['auto', 'sqrt', 'log2'],
                  'model__n_estimators' : [x for x in range(1, 1000, 50)]}]


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_grid = {
    'model__n_estimators': n_estimators,
               'model__max_features': max_features,
               'model__max_depth': max_depth,
               'model__min_samples_split': min_samples_split,
               'model__min_samples_leaf': min_samples_leaf,
               'model__bootstrap': bootstrap}


estimator = pipe

# for score in scores:
grid(Xtrain = X,
            ytrain = y,
            estimator = pipe,
            params_grid = param_grid,
            scores=scores,
            cvs = cv,
            cores=core,
            verb=verbos)

print(score)


estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# Lasso
#==============================================================================
from sklearn.linear_model import Lasso
model = Lasso()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__alpha' : [x*0.1 for x in range(1,10)],
                  'model__max_iter' : [x for x in range(50, 10000, 50)],
                  'model__selection' : ['cyclic','random']}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# RandomForestRegressor
#==============================================================================
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__max_depth' : [x for x in range(1, 10)],
                  'model__max_features' : ['auto', 'sqrt', 'log2'],
                  'model__n_estimators' : [x for x in range(1, 1000, 50)]}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)






models = [
    RandomForestRegressor(random_state=1, n_jobs=-1), 
#     Lasso(random_state=1)
]


# CatBoostRegressor(depth=8),


models = [GradientBoostingRegressor(subsample=0.8,learning_rate=0.4, random_state=500),
#           RandomForestRegressor(max_depth=250, min_samples_split=14,min_samples_leaf =3, random_state=1, n_jobs = -1),
          RandomForestRegressor(n_estimators = 1000, max_depth=250, min_samples_split=5, random_state=0, n_jobs = -1),
          RandomForestRegressor(n_estimators = 50, max_depth=100, min_samples_split=400, random_state=0, n_jobs = -1),
          Lasso(alpha= 4.5)]

# from tensorflow.keras import models
from tensorflow.keras import layers

def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)])
    
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

temp_list = [x for x in all_df_now if not 'Robot' in x]

# Making list of df for conct before training
# This is different form list of srtings, as this is a list of actual dataframes
df_list = []
for x in temp_list:
    df_list.append(locals()[x])

df_ = pd.concat(df_list)

X = df_[training_features]
y = df_[target_features].values.ravel()
groups = df_[group_feature].values.ravel()

gkf = list(GroupKFold(n_splits=6).split(X, y, groups))
# gkf = list(StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=1).split(X, y, groups))

#     Getting scores using cross_val_score
for model in models:
    print(model)
    importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp = training_gkf_std(
        model, X, y, gkf)
    if importances is not None:
        plot_feat_imp(importances,
                      model,
                      training_features,
                      threshold=threshold_all)

    save_results(model=model,
                 agg_method=agg_method,
                 train_field='all_mix',
                 test_field='all_mix',
                 training_features=training_features,
                 importances=importances,
                 RMSE_test=RMSE_test_temp,
                 RMSE_train=RMSE_train_temp,
                 R2_test=R2_test_temp,
                 R2_train=R2_train_temp,
                 GKF_CV=GKF_CV_temp)
del (temp_list, df_list, df_, X, y, groups, gkf)
del (importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp, model)

# Iterating through all possible permutations of the fields dataset
for df in all_df_now:
    df_ = locals()[df].copy()

    X = df_[training_features]
    y = df_[target_features].values.ravel()
    groups = df_[group_feature].values.ravel()

    gkf = list(GroupKFold(n_splits=3).split(X, y, groups))
    print(df)
    #     Getting scores using cross_val_score
    for model in models:
        print(df)
        importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp = training_gkf_std(
            model, X, y, gkf)
        if importances is not None:
            plot_feat_imp(importances,
                          model,
                          training_features,
                          threshold=threshold_all)

        save_results(model=model,
                     agg_method=agg_method,
                     train_field=df,
                     test_field=df,
                     training_features=training_features,
                     importances=importances,
                     RMSE_test=RMSE_test_temp,
                     RMSE_train=RMSE_train_temp,
                     R2_test=R2_test_temp,
                     R2_train=R2_train_temp,
                     GKF_CV=GKF_CV_temp)
    del (df, df_, X, y, groups, gkf)
    del (importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp, model)

# # Iterating through all possible permutations of the fields dataset

# for i in itertools.permutations(all_df_now, 2):
#     train_df = locals()[i[0]].copy()
#     test_df = locals()[i[1]].copy()
    
    
#     X_train = train_df[training_features]
#     y_train = train_df[target_features].values.ravel()
#     X_test = test_df[training_features]
#     y_test = test_df[target_features].values.ravel()
    
#     # Getting scores using cross_val_score
#     for model in models:
#         print('Training: ', i[0],'Test: ', i[1], ' : ', model)
#         importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp = training_regr(
#             model, X_train, y_train, X_test, y_test)
#         if importances is not None:
#             plot_feat_imp(importances, model, training_features, threshold=threshold_all, sort_feat=True)
            
#         save_results(model=model,
#              agg_method=agg_method,
#              train_field=i[0],
#              test_field=i[1],
#              training_features=training_features,
#              importances=importances,
#              RMSE_test=RMSE_test_temp,
#              RMSE_train=RMSE_train_temp,
#              R2_test=R2_test_temp,
#              R2_train=R2_train_temp,
#              GKF_CV=GKF_CV_temp)
#     del (i, train_df, test_df, X_train, y_train, X_test, y_test)
#     del (importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp, model)

# Iterating through all possible permutations of the fields dataset
for df in all_df_now:
    if 'Robot' not in df:
        temp_list = [
            x for x in all_df_now if not 'Robot' in x if not df in x
        ]
        print(df, temp_list)

        # Making list of df for conct before training
        # This is different form list of srtings, as this is a list of actual dataframes
        train_df_list = []
        for x in temp_list:
            train_df_list.append(locals()[x])

        train_df = pd.concat(train_df_list)
        test_df = locals()[df].copy()

        X_train = train_df[training_features]
        y_train = train_df[target_features].values.ravel()
        X_test = test_df[training_features]
        y_test = test_df[target_features].values.ravel()

        # Getting scores using cross_val_score
        for model in models:
            print('Training: All  ', 'Test: ', df, ' : ', model)
            importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp = training_regr(
                model, X_train, y_train, X_test, y_test)
#             if importances is not None:
#                 plot_feat_imp(importances,
#                               model,
#                               training_features,
#                               threshold=threshold_all,
#                               sort_feat=sorted_all)
            save_results(model=model,
                         agg_method=agg_method,
                         train_field=temp_list,
                         test_field=df,
                         training_features=training_features,
                         importances=importances,
                         RMSE_test=RMSE_test_temp,
                         RMSE_train=RMSE_train_temp,
                         R2_test=R2_test_temp,
                         R2_train=R2_train_temp,
                         GKF_CV=GKF_CV_temp)
        del (df, temp_list, train_df_list, train_df, test_df, X_train, y_train, X_test, y_test)
        del (importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp, model)

train_str_list, test_str_list = list_test_train_df(all_df_now,
                                                   train_field = 'Staur', 
                                                   test_field = 'Vollebekk', 
                                                   year = 'all')

train_df_list = []
test_df_list = []
for x in train_str_list:
    train_df_list.append(locals()[x])
for x in test_str_list:
    test_df_list.append(locals()[x])

train_df = pd.concat(train_df_list)
test_df = pd.concat(test_df_list)

X_train = train_df[training_features]
y_train = train_df[target_features].values.ravel()
X_test = test_df[training_features]
y_test = test_df[target_features].values.ravel()

# Getting scores using cross_val_score
for model in models:
    importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp = training_regr(
        model, X_train, y_train, X_test, y_test)
    if importances is not None:
        plot_feat_imp(importances,
                      model,
                      training_features,
                      threshold=threshold_all,
                      sort_feat=sorted_all)
    save_results(model=model,
                 agg_method=agg_method,
                 train_field=train_str_list,
                 test_field=test_str_list,
                 training_features=training_features,
                 importances=importances,
                 RMSE_test=RMSE_test_temp,
                 RMSE_train=RMSE_train_temp,
                 R2_test=R2_test_temp,
                 R2_train=R2_train_temp,
                 GKF_CV=GKF_CV_temp)
del (train_str_list, test_str_list, train_df_list, test_df_list, train_df, test_df, X_train, y_train, X_test, y_test)
del (importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp, model)

train_str_list, test_str_list = list_test_train_df(all_df_now,
                                                   train_field = 'Vollebekk', 
                                                   test_field = 'Staur', 
                                                   year = 'all')

train_df_list = []
test_df_list = []
for x in train_str_list:
    train_df_list.append(locals()[x])
for x in test_str_list:
    test_df_list.append(locals()[x])

train_df = pd.concat(train_df_list)
test_df = pd.concat(test_df_list)

X_train = train_df[training_features]
y_train = train_df[target_features].values.ravel()
X_test = test_df[training_features]
y_test = test_df[target_features].values.ravel()

# Getting scores using cross_val_score
for model in models:
    importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp = training_regr(
        model, X_train, y_train, X_test, y_test)
    if importances is not None:
        plot_feat_imp(importances,
                      model,
                      training_features,
                      threshold=threshold_all,
                      sort_feat=sorted_all)
    save_results(model=model,
                 agg_method=agg_method,
                 train_field=train_str_list,
                 test_field=test_str_list,
                 training_features=training_features,
                 importances=importances,
                 RMSE_test=RMSE_test_temp,
                 RMSE_train=RMSE_train_temp,
                 R2_test=R2_test_temp,
                 R2_train=R2_train_temp,
                 GKF_CV=GKF_CV_temp)
del (train_str_list, test_str_list, train_df_list, test_df_list, train_df, test_df, X_train, y_train, X_test, y_test)
del (importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp, model)

train_str_list, test_str_list = list_test_train_df(all_df_now,
                                                   train_field = 'Staur', 
                                                   test_field = 'Vollebekk', 
                                                   year = '2020')

train_df_list = []
test_df_list = []
for x in train_str_list:
    train_df_list.append(locals()[x])
for x in test_str_list:
    test_df_list.append(locals()[x])

train_df = pd.concat(train_df_list)
test_df = pd.concat(test_df_list)

X_train = train_df[training_features]
y_train = train_df[target_features].values.ravel()
X_test = test_df[training_features]
y_test = test_df[target_features].values.ravel()

# Getting scores using cross_val_score
for model in models:
    importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp = training_regr(
        model, X_train, y_train, X_test, y_test)
    if importances is not None:
        plot_feat_imp(importances,
                      model,
                      training_features,
                      threshold=threshold_all,
                      sort_feat=sorted_all)
    save_results(model=model,
                 agg_method=agg_method,
                 train_field=train_str_list,
                 test_field=test_str_list,
                 training_features=training_features,
                 importances=importances,
                 RMSE_test=RMSE_test_temp,
                 RMSE_train=RMSE_train_temp,
                 R2_test=R2_test_temp,
                 R2_train=R2_train_temp,
                 GKF_CV=GKF_CV_temp)
del (train_str_list, test_str_list, train_df_list, test_df_list, train_df, test_df, X_train, y_train, X_test, y_test)
del (importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp, model)

train_str_list, test_str_list = list_test_train_df(all_df_now,
                                                   train_field = 'Vollebekk', 
                                                   test_field = 'Staur', 
                                                   year = '2020')

train_df_list = []
test_df_list = []
for x in train_str_list:
    train_df_list.append(locals()[x])
for x in test_str_list:
    test_df_list.append(locals()[x])

train_df = pd.concat(train_df_list)
test_df = pd.concat(test_df_list)

X_train = train_df[training_features]
y_train = train_df[target_features].values.ravel()
X_test = test_df[training_features]
y_test = test_df[target_features].values.ravel()

# Getting scores using cross_val_score
for model in models:
    importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp = training_regr(
        model, X_train, y_train, X_test, y_test)
    if importances is not None:
        plot_feat_imp(importances,
                      model,
                      training_features,
                      threshold=threshold_all,
                      sort_feat=sorted_all)
    save_results(model=model,
                 agg_method=agg_method,
                 train_field=train_str_list,
                 test_field=test_str_list,
                 training_features=training_features,
                 importances=importances,
                 RMSE_test=RMSE_test_temp,
                 RMSE_train=RMSE_train_temp,
                 R2_test=R2_test_temp,
                 R2_train=R2_train_temp,
                 GKF_CV=GKF_CV_temp)
del (train_str_list, test_str_list, train_df_list, test_df_list, train_df, test_df, X_train, y_train, X_test, y_test)
del (importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp, model)

train_str_list, test_str_list = list_test_train_df(all_df_now,
                                                   train_field = 'Staur', 
                                                   test_field = 'Vollebekk', 
                                                   year = '2019')

train_df_list = []
test_df_list = []
for x in train_str_list:
    train_df_list.append(locals()[x])
for x in test_str_list:
    test_df_list.append(locals()[x])

train_df = pd.concat(train_df_list)
test_df = pd.concat(test_df_list)

X_train = train_df[training_features]
y_train = train_df[target_features].values.ravel()
X_test = test_df[training_features]
y_test = test_df[target_features].values.ravel()

# Getting scores using cross_val_score
for model in models:
    importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp = training_regr(
        model, X_train, y_train, X_test, y_test)
    if importances is not None:
        plot_feat_imp(importances,
                      model,
                      training_features,
                      threshold=threshold_all,
                      sort_feat=sorted_all)
    save_results(model=model,
                 agg_method=agg_method,
                 train_field=train_str_list,
                 test_field=test_str_list,
                 training_features=training_features,
                 importances=importances,
                 RMSE_test=RMSE_test_temp,
                 RMSE_train=RMSE_train_temp,
                 R2_test=R2_test_temp,
                 R2_train=R2_train_temp,
                 GKF_CV=GKF_CV_temp)
del (train_str_list, test_str_list, train_df_list, test_df_list, train_df, test_df, X_train, y_train, X_test, y_test)
del (importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp, model)

train_str_list, test_str_list = list_test_train_df(all_df_now,
                                                   train_field = 'Vollebekk', 
                                                   test_field = 'Staur', 
                                                   year = '2019')

train_df_list = []
test_df_list = []
for x in train_str_list:
    train_df_list.append(locals()[x])
for x in test_str_list:
    test_df_list.append(locals()[x])

train_df = pd.concat(train_df_list)
test_df = pd.concat(test_df_list)

X_train = train_df[training_features]
y_train = train_df[target_features].values.ravel()
X_test = test_df[training_features]
y_test = test_df[target_features].values.ravel()

# Getting scores using cross_val_score
for model in models:
    importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp = training_regr(
        model, X_train, y_train, X_test, y_test)
    if importances is not None:
        plot_feat_imp(importances,
                      model,
                      training_features,
                      threshold=threshold_all,
                      sort_feat=sorted_all)
    save_results(model=model,
                 agg_method=agg_method,
                 train_field=train_str_list,
                 test_field=test_str_list,
                 training_features=training_features,
                 importances=importances,
                 RMSE_test=RMSE_test_temp,
                 RMSE_train=RMSE_train_temp,
                 R2_test=R2_test_temp,
                 R2_train=R2_train_temp,
                 GKF_CV=GKF_CV_temp)
del (train_str_list, test_str_list, train_df_list, test_df_list, train_df, test_df, X_train, y_train, X_test, y_test)
del (importances, RMSE_test_temp, RMSE_train_temp, R2_test_temp, R2_train_temp, GKF_CV_temp, model)



results_csv = pd.read_csv(export_path+'results_org.csv')
res_df = results_csv[['Aggregation_method','Train_field', 'Test_field', 'RMSE_test', 'RMSE_train',
       'R2_test', 'R2_train']]

res_simp = res_df[res_df.Aggregation_method == 'Simpsons']
res_simp.drop(['Aggregation_method'], axis=1, inplace=True)
res_simp

plot_res_df = np.array(res_simp.iloc[49:-1,4:])
plot_res_df = plot_res_df.astype(np.float)
plot_res_df

import numpy  as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matrix = plot_res_df

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xpos = [range(matrix.shape[0])]
ypos = [range(matrix.shape[1])]
xpos, ypos = np.meshgrid(xpos, ypos)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = matrix.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz,  zsort='average')

plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x, y = np.random.rand(2, 100) * 4
hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()

y

hist

threshold_all = 'top_25'
sorted_all = True
agg_method = 'Simpsons'
# agg_method = 'Trapezoid'
# training_features = base_indices + spectral_indices + environment_var
# training_features = base_indices + spectral_indices + weather_features
training_features =  spectral_indices + weather_features
# training_features = spectral_indices

target_features = ['GrainYield']

group_feature = ['Name']

if agg_method == 'Simpsons':
    all_df_now = all_df_simps.copy()
elif agg_method == 'Trapezoid': 
    all_df_now = all_df_trapz.copy()

temp_list = [x for x in all_df_now if not 'Robot' in x]

# Making list of df for conct before training
# This is different form list of srtings, as this is a list of actual dataframes
df_list = []
for x in temp_list:
    df_list.append(locals()[x])

df_ = pd.concat(df_list)

X = df_[training_features].values
y = df_[target_features].values
groups = df_[group_feature].values.ravel()

gkf = list(GroupKFold(n_splits=6).split(X, y, groups))


scores = ['neg_root_mean_squared_error', 'r2']
cv = 5
core = 6
verbos = 5

scores = ['neg_root_mean_squared_error', 'r2']
cv = 5
core = 6
verbos = 5

#==============================================================================
# RandomForestRegressor
#==============================================================================
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_grid = {
    'model__n_estimators': n_estimators,
#                'model__max_features': max_features,
#                'model__max_depth': max_depth,
#                'model__min_samples_split': min_samples_split,
#                'model__min_samples_leaf': min_samples_leaf,
               'model__bootstrap': bootstrap}


estimator = pipe

# for score in scores:
# grid(Xtrain = X,
#             ytrain = y,
#             estimator = pipe,
#             params_grid = param_grid,
#             scores=scores,
#             cvs = cv,
#             cores=core,
#             verb=verbos)
# print(score)

#==============================================================================
# GradientBoostingRegressor
#==============================================================================
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
# param_grid   =  [{'model__loss' : ['ls', 'lad', 'huber', 'quantile'],
# #                   'model__learning_rate' : [0.001, 0.01, 0.1, 1],
# #                   'model__n_estimators' : range(0,500, 100),
                  
# #                   'model__max_depth':range(5,16,2), 
# #                   'model__min_samples_split':range(200,1100, 200), # 2100
# #                   'model__min_samples_leaf':range(30,71,10),
#                   'model__max_features':range(7,20,2),
#                   'model__subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}]

param_grid   =  [{'model__loss' : ['huber'],
#                   'model__learning_rate' : [0.001, 0.01, 0.1, 1],
#                   'model__n_estimators' : range(0,500, 100),
                  
                  'model__max_depth':range(5,16,2), 
#                   'model__min_samples_split':range(2,5), # 2100
#                   'model__min_samples_leaf':range(1,2),
#                   'model__max_features':range(5,6),
                  'model__subsample':[0.7,0.8]}]
# pipe.get_params()
estimator = pipe

for score in scores:
    grid(Xtrain = X.values,
                ytrain = y.values,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# Lasso
#==============================================================================
from sklearn.linear_model import Lasso
model = Lasso()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__alpha' : [x*0.1 for x in range(1,10)],
                  'model__max_iter' : [x for x in range(50, 10000, 50)],
                  'model__selection' : ['cyclic','random']}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# Ridge
#==============================================================================
from sklearn.linear_model import Ridge
model = Ridge()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__alpha' : [x*1. for x in range(1,10)],
                  'model__solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# ElasticNet
#==============================================================================
from sklearn.linear_model import ElasticNet
model = ElasticNet()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__alpha' : [x*0.1 for x in range(1,10)],
                  'model__max_iter' : [x for x in range(50, 10000, 50)],
                  'model__l1_ratio' : [x*0.1 for x in range(1,10)]}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# OrthogonalMatchingPursuit
#==============================================================================
from sklearn.linear_model import OrthogonalMatchingPursuit
model = OrthogonalMatchingPursuit()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__fit_intercept' : [True, False],
                  'model__n_nonzero_coefs' : [x for x in range(1,10)]}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# BayesianRidge
#==============================================================================
from sklearn.linear_model import BayesianRidge
model = BayesianRidge()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__n_iter' : [x for x in range(5, 150, 10)],
                  'model__alpha_1' : [1.0],
                  'model__alpha_2' : [x*0.000001 for x in [1,10,100,1000,10000,100000,1000000]],
                  'model__lambda_1' : [x*0.000001 for x in [1,10,100,1000,10000,100000,1000000]],
                  'model__lambda_2' : [1.0]}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# ARDRegression
#==============================================================================
from sklearn.linear_model import ARDRegression
model = ARDRegression()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__n_iter' : [x for x in range(5, 150, 10)],
                  'model__alpha_1' : [1.0],
#                       'model__alpha_2' : [x*0.000001 for x in [1,10,100,1000,10000,100000,1000000]],
                  'model__lambda_1' : [0.01],
                  'model__lambda_2' : [x*0.000001 for x in [1,10,100,1000,10000,100000,1000000]],
                  'model__verbose' : [True]}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# RANSACRegressor
#==============================================================================
from sklearn.linear_model import RANSACRegressor
model = RANSACRegressor()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__min_samples' : [x/.1 for x in range(1, 10)],
                  'model__max_trials' : [x for x in range(1, 500,50)],
                  'model__loss' : ['absolute_loss', 'squared_loss']}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# TheilSenRegressor
#==============================================================================
# from sklearn.linear_model import TheilSenRegressor
# model = TheilSenRegressor()
# sc = StandardScaler()
# pipe = Pipeline(steps=[('sc', sc), ('model', model)])
# param_grid   =  [{'model__max_subpopulation' : [x*0.000001 for x in [1,10,100,1000,10000,100000,1000000]],
#                   'model__n_subsamples' : [x for x in range(9, 1300,50)],
#                   'model__max_iter' :  [x for x in range(50, 1000, 50)]}]
# estimator = pipe

# for score in scores:
#     grid(Xtrain = X,
#                 ytrain = y,
#                 estimator = pipe,
#                 params_grid = param_grid,
#                 scores=score,
#                 cvs = cv,
#                 cores=core,
#                 verb=verbos)
#     print(score)

#==============================================================================
# HuberRegressor
#==============================================================================
from sklearn.linear_model import HuberRegressor
model = HuberRegressor()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__epsilon' : [x/.01 for x in range(100, 200, 5)],
                  'model__alpha' : [x*0.000001 for x in [1,10,100,1000,10000,100000,1000000]]}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# DecisionTreeRegressor
#==============================================================================
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__max_depth' : [None]+[x for x in range(1, 100,5)],
                  'model__min_samples_leaf' : [x for x in range(1, 50,5)],
                  'model__min_samples_split' : [2]+[x for x in range(1, 50,5)],
                  'model__max_features' : [x for x in range(1, 10)]}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# GaussianProcessRegressor
#==============================================================================
from sklearn.gaussian_process import GaussianProcessRegressor
model = GaussianProcessRegressor()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__kernel' : [None]+['rbf', 'sigmoid',  'linear', 'poly'],
                  'model__alpha' : [x*0.0000000001 for x in [1,10,100,1000,10000,100000,1000000]]}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

#==============================================================================
# KNeighborsRegressor
#==============================================================================
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__n_neighbors' : [x for x in range(1, 100,5)],
                  'model__weights' : ['uniform', 'distance'],
                  'model__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'model__leaf_size' : [x for x in range(10, 50, 5)]}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)

# #==============================================================================
# # RadiusNeighborsRegressor
# #==============================================================================
# from sklearn.neighbors import RadiusNeighborsRegressor
# model = RadiusNeighborsRegressor()
# sc = StandardScaler()
# pipe = Pipeline(steps=[('sc', sc), ('model', model)])
# param_grid   =  [{'model__radius' : [x*1. for x in range(1, 10)],
#                   'model__weights' : ['uniform', 'distance'],
#                   'model__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
#                   'model__p' : [x for x in range(1, 10)]}]
# estimator = pipe

# for score in scores:
#     grid(Xtrain = X,
#                 ytrain = y,
#                 estimator = pipe,
#                 params_grid = param_grid,
#                 scores=score,
#                 cvs = cv,
#                 cores=core,
#                 verb=verbos)
#     print(score)

# #==============================================================================
# # SVR
# #==============================================================================
# from sklearn.svm import SVR
# model = SVR()
# sc = StandardScaler()
# pipe = Pipeline(steps=[('sc', sc), ('model', model)])
# param_grid   =  [{'model__radius' : [x*1. for x in range(1, 10)],
#                   'model__weights' : ['uniform', 'distance'],
#                   'model__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
#                   'model__p' : [x for x in range(1, 10)]}]
# estimator = pipe

# for score in scores:
#     grid(Xtrain = X,
#                 ytrain = y,
#                 estimator = pipe,
#                 params_grid = param_grid,
#                 scores=score,
#                 cvs = cv,
#                 cores=core,
#                 verb=verbos)
#     print(score)

#==============================================================================
# RandomForestRegressor
#==============================================================================
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
sc = StandardScaler()
pipe = Pipeline(steps=[('sc', sc), ('model', model)])
param_grid   =  [{'model__max_depth' : [x for x in range(1, 10)],
                  'model__max_features' : ['auto', 'sqrt', 'log2'],
                  'model__n_estimators' : [x for x in range(1, 1000, 50)]}]
estimator = pipe

for score in scores:
    grid(Xtrain = X,
                ytrain = y,
                estimator = pipe,
                params_grid = param_grid,
                scores=score,
                cvs = cv,
                cores=core,
                verb=verbos)
    print(score)


# t_end = time.time()
# tt = t_end - t_start
# time_taken.append(tt)
# print('Total time complete: ', (tt) / 60, 'minutes')



# PERMUTATION

from sklearn.inspection import permutation_importance


result = permutation_importance(gs_xgb_fitted, X_test, y_test, n_repeats=100, random_state=0)

# ==================================
# Feature selection
# ===================================

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# define feature selection
fs = SelectKBest(score_func=f_regression, k=10)
# apply feature selection
X_selected = fs.fit_transform(X, y)
print(X_selected.shape)

# Plot importances
fig, ax = plt.subplots(figsize=(25, 25))
ind = indices = np.argsort(result.importances_mean)[::-1]
plt.barh(X_test.columns, result.importances_mean[ind])
plt.show()


from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average = 'macro'),
           'recall': make_scorer(recall_score, average = 'macro'),
           'f1': make_scorer(f1_score, average = 'macro')}
grid_search_rfc = GridSearchCV(rfc, param_grid = grid_values, scoring = scoring, refit='f1')
grid_search_rfc.fit(x_train, y_train)

grid_search_rfc.best_params_
grid_search_rfc.cv_results_

# cv_results[‘mean_test_<metric_name>’]
grid_search_rfc.cv_results_['mean_test_recall']


from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")


