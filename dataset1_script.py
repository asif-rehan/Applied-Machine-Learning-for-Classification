'''
Created on Nov 25, 2014

@author: asr13006
'''
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pandas.io.pytables import read_hdf, to_hdf
from sklearn.decomposition.pca import PCA
from classification import feat_combo_classifier as FCC
import numpy as np
from matplotlib.pyplot import savefig
import pandas as pd
from matplotlib import pyplot
from sklearn import preprocessing
from scipy.io.matlab.mio import loadmat
from sklearn.linear_model.sgd_fast import Regression


def two_class_sampled(df, sample_proportion):
    classwise_sampled = {}
    for i in [0,1]:
        class_ith = df[df['y2'] == i]
        random_rows = np.random.choice(class_ith.index.values, 
                                       sample_proportion*class_ith.shape[0])
        classwise_sampled[i] = class_ith.ix[random_rows]
    del df
    return classwise_sampled

def equal_prior_sampled_all(filenum):
    columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'y1', 'y2']
    df = pd.DataFrame()
    for i in columns:
        df[i] = loadmat('dataset{}_costDist_v3'.format(filenum))[i][0]
    min_max_scaler = preprocessing.MinMaxScaler(copy=False)
    scaled_all_col = min_max_scaler.fit_transform(df)
    del df
    scaled_all_col = pd.DataFrame(scaled_all_col, columns=columns)
    class1 = scaled_all_col[scaled_all_col['y2'] == 1]
    class1_len, class0_len = scaled_all_col['y2'].value_counts()
    drop_rows_1 = np.random.choice(class1.index.values, 
                                   size=class1_len-class0_len,
                                   replace=False)
    del class1 
    scaled_all_col.drop(drop_rows_1, inplace=True)
    scaled_all_col = scaled_all_col.set_index(
                                [[i for i in range(2*class0_len)]])
    scaled_all_col.to_hdf('DO_NOT_CHANGE_mat1{}_equal_prior_11255549x2.h5'.
                          format(filenum), 'data')
'''
def regression_data_file_maker(sample_percent):
    regression_data = pd.DataFrame(columns=['f1', 'f2', 'f3', 'f4', 'f5','y1']) 
    for filenum, sample_size_a_class  in [(1, 928828), 
                                            (2, 11255549), 
                                            (3,804548)]:
        data = read_hdf('DO_NOT_CHANGE_mat{}_equal_prior_{}x2.h5'.format(int(filenum), sample_size_a_class), 'data')
        drop_rows = np.random.choice(data.index.values, 
                                       size=(sample_percent*2*
                                                sample_size_a_class),
                                       replace=False)
        data.drop(drop_rows, inplace=True)
        regression_data.append(data)
    regression_data.to_hdf('DO_NOT_CHANGE_regression_data_10%.h5', 'data')

regression_data_file_maker(0.1)
'''      

def combo_scatterplot(classwise_sampled, columns, plot_name):
    fcc = FCC(classwise_sampled)
    num_of_column = len(columns)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for combo in fcc.combo_creator(columns[0:num_of_column-1]):
        if len(combo) < 4:
            fig = plt.figure()
            if len(combo) == 2:
                ax = fig.add_subplot(111)
            if len(combo) == 3:
                ax = fig.add_subplot(111, projection='3d')
            for classnum, marker, c in ((0, 'o', 'b'), (1, '^', 'r')):
                sample_c_i = classwise_sampled[classnum]
                plot_data = sample_c_i[combo]
                if len(combo) == 1:
                    pyplot.hist(np.array(plot_data[:][combo[0]]), alpha=0.5, 
                                label='class{}'.format(classnum))
                    pyplot.legend(loc='upper right')
                if len(combo) == 2:
                    plt.scatter(plot_data[:][combo[0]], 
                               plot_data[:][combo[1]], c=c, marker=marker)
                    ax.set_xlabel(combo[0])
                    ax.set_ylabel(combo[1])
                if len(combo) == 3:
                    ax.scatter(plot_data[:][combo[0]], 
                               plot_data[:][combo[1]],
                               plot_data[:][combo[2]], c=c, marker=marker)
                    ax.set_xlabel(combo[0])
                    ax.set_ylabel(combo[1])
                    ax.set_zlabel(combo[2])
            plot_title = "{0}_{1}".format(plot_name, combo)  
            plt.title(plot_title)
            savefig(save_path+'/'+plot_title)   
            #plt.show()
        else:
            break
    return None
    
    
    
    
    
if __name__ == '__main__':
    cwd = r'C:\Users\asr13006\Desktop\ECE 6141 Neural Network\Homework_Shared\Dataset 1\dataset1 newfiles'.replace("\\", '/')
    os.chdir(cwd)
    save_path = r"C:\Users\asr13006\Google Drive\UConn MS\Fall 2014\ECE 6141 Neural Network\Homework_Shared\Dataset1 analysis outputs\combo_scatter\combo_scatter2".replace('\\', '/')
    
    df = read_hdf('DO_NOT_CHANGE_eq_prior_final.h5', 'data')
    columns = df.keys()
    
    ###########################################################################
    #    complete dataset
    ###########################################################################
    
    
    
    ###########################################################################
    #    unscaled plots
    ###########################################################################
    
    #classwise_sampled = two_class_sampled(df, 0.01)
    #combo_scatterplot(classwise_sampled, 
    #                  ['f1', 'f2', 'f3','f4', 'f5', 'y2'],
    #                  'Scatter Plot')
    ###########################################################################
    #    PCA (SVD, WHITENING) plots
    ###########################################################################
    
    '''
    pca = PCA(3, whiten=True)
    pca.fit(df[df.columns[0:5]])
    
    transform = pca.transform(df[:][df.columns[:5]])
    df_transform = pd.DataFrame(transform)
    df_transform['y2'] = np.array(df[:]['y2'])
    
    classwise_sampled_transform = two_class_sampled(df_transform, 0.01)
    combo_scatterplot(classwise_sampled_transform, [0,1,2,'y2'],
                      'PCA Scatter Plot')
    
    '''
    """
    ###########################################################################
    #    standardize plots
    ###########################################################################
    
    x_columns = list(df.columns[:5])
    scaled_X = pd.DataFrame(preprocessing.scale(df[:][x_columns]), 
                                    columns = x_columns)
    df = df.set_index([[i for i in range(len(df))]])
    df[x_columns] = scaled_X[x_columns]
    
    classwise_sampled_transform = two_class_sampled(df, 0.01)
    combo_scatterplot(classwise_sampled_transform,  
                      ['f1', 'f2', 'f3','f4', 'f5', 'y2'],
                      'Standardized Scatter Plot')
    
    ###########################################################################
    #    minmax scaled plots
    ###########################################################################
    x_columns = list(df.columns[:5])
    mmax_scld_X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(
                                            df[:][x_columns]),
                                            columns = x_columns)
    df = df.set_index([[i for i in range(len(df))]])
    df[x_columns] = mmax_scld_X[x_columns]
    df.to_hdf('DO_NOT_CHANGE_minmax_scaled.h5', 'data')
    
    """
         
    ###########################################################################
    # save_pca_as_hdf(df)   
    ###########################################################################
    pca5 = PCA(5, whiten=True)
    pca5.fit(df)
    pca5.explained_variance_ratio_
    #output: array([ 0.8992,  0.0971,  0.0037,  0.    ,  0.    ])
    
    pca = PCA(3, whiten=True)
    pca.fit(df)
    pca.explained_variance_ratio_
    pca.transform(df)
    pca_df = pd.DataFrame(pca.transform(df), columns=[0,1,2])
    pca_df['y2'] = 
    pca_df.to_hdf(cwd+'/'+'DO_NOT_CHANGE_PCA.h5', 'data')
    
    ###########################################################################
    #    full_data_file_with_['f1', 'f2', 'f3', 'f4', 'f5', 'y1', 'y2']
    #    from 'DO_NOT_CHANGE_mat1{}_equal_prior_{}x2.h5' 
    ###########################################################################
    
            