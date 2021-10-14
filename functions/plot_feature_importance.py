import numpy as np
import pandas as pd

import os
import math
import datetime
import numpy as np
import pandas as pd
from copy import copy

# Dictionaries
import json
from pprint import pprint

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feat_imp(X_all = X, y_all = y, model_pipe = current_model, train_feat = training_features):
    # Plotting feature importance
    # Create arrays from feature importance and feature names
    model_pipe.fit(X_all, y_all)
    feature_importance = np.array(model_pipe.steps[1][1].feature_importances_)
    feature_names = train_feat.copy()
    model_name =  str(model_pipe).split('(')[-2].split('  ')[-1]

    # Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
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
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    # plt.savefig('Data/feature_importance_'+model_type+'.pdf',dpi=500, bbox_inches='tight')
    plt.show()