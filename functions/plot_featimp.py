import os
import textwrap
import numpy as np
import pandas as pd
import random
# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# from matplotlib.backends.backend_pdf import PdfPages

# # Create plots folder if not exists already
# os.makedirs(plots_export_path, exist_ok=True)

# pdf = PdfPages(plots_export_path+'feat_imp.pdf')

def plot_feat_imp(feature_importance, model, train_feat, feature_group = 0, threshold='all', sort_feat=True, 
                  show_plot = True, save_plot = False,  export_path = './', save_suffix = ' '):
    
    # threshold =  percentage of max(features_importance) or 'all' or top_x number of features
    # save_suffix = suffix to follow the saved file name and also in title of plot
    # NOTEEE = save_suffix cannot have INTTTTTTTTTT or numbers in it
    # Plotting feature importance
    # Create arrays from feature importance and feature names

    feature_names = train_feat.copy()
    model_name =  str(model).split('(')[0]
    model_name_full = ''.join([ c if c.isalnum() else "_" for c in str(model).replace(" ", "") ])
    
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

    if show_plot:
        #Define size of bar plot
        plt.figure(figsize=(10,7))

        # Fixing the problem where when all x labels are not of same size, they are not positioned correctly and
        # move out of the plot bottom area
        # Even spacing of rotated axis labels in matplotlib and/or seaborn
        # https://stackoverflow.com/questions/43618423/even-spacing-of-rotated-axis-labels-in-matplotlib-and-or-seaborn
        # print(feat_imp_df['feature_names'])
        # print(type(feat_imp_df['feature_names']))
        # print([len(i) for i in feat_imp_df['feature_names']])
        # print(np.mean([len(i) for i in feat_imp_df['feature_names']]))
        
        mean_length = np.mean([len(i) for i in feat_imp_df['feature_names']])
        columns = ["\n".join(textwrap.wrap(i,int(mean_length))) for i in feat_imp_df['feature_names']]
        
        #Plot Searborn bar chart
        ax = sns.barplot(y=feat_imp_df['feature_importance'], x=feat_imp_df['feature_names'], palette = 'winter'  )
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=30,
                           ha="right",
                           rotation_mode='anchor')

        
        #Add chart labels
        plt.title(model_name + ' ' + save_suffix + '_feat_grp_' + str(feature_group) + ' Feature Importance')
        plt.xlabel('Feature Names')
        plt.ylabel('Feature Importance')
       
        plt.tight_layout()


        # Saving plots
        export_plots = export_path+'/Feature_Importance_Feat_Grp_'+str(feature_group)+'/'+model_name_full+'/'
        os.makedirs(export_plots, exist_ok=True)
        if save_plot:
            # There was peoblem with saving plots in a loop where 66-33 ratio was used for same df
            # So, using try method to avoid errors and saving plots with slightly different names
            filename = '/'+str(save_suffix)+'-feat_grp_'+str(feature_group)+'.jpg'
            alt_filename = '/'+str(save_suffix)+'g'+str(feature_group)+'.jpg'
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

        #     plt.savefig(export_plots+col+feature_importance_'+model_name+'.pdf',dpi=500, bbox_inches='tight')

        plt.show()