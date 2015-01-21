'''
Created on Nov 10, 2014

@author: asr13006
'''

import os
import time
import sys
import scipy.io
import pandas as pd
import numpy as np
from classification import feat_combo_classifier
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition.pca import PCA
from numpy import argmin
from pandas.io.pytables import HDFStore

cwd = r'path_to_datafile'.replace("\\", '/')
os.chdir(cwd)


def eq_prior_sampler_from_hdf(filenum, prop_sampled, HDF_file_name):
    """
    HDF_file_name = "mat{0}.h5".format(i) for i in [1,2,3]
    prop_sampled = proportion of total data to be sampled from all 3 files
    """
    columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'y2']
    cls_col = 'y2'
    #hdf = HDFStore('{}.h5'.format(HDF_file_name))
    eq_prior_sampled = pd.DataFrame(columns=columns)
    print len(eq_prior_sampled)
    for i in filenum:
        df = HDFStore("mat{0}.h5".format(i), 'r')['mat{}'.format(i)]
                
        class1_mat = df[df['y2'] == 1]
        class0_mat = df[df['y2'] == 0]        
        del df
        sample_limit = min(len(class0_mat),len(class1_mat))
        #smaller_class = argmin(len(class0_mat),len(class1_mat))
        #here class0 has smaller sample size
        
        #automatic weighting could be added in future
        #class1_proportion = float(len(class1_mat)/
        #                          (len(class0_mat)+len(class0_mat)))
        #class0_proportion = 1 - class1_proportion
        class1_to_class0_ratio = float(len(class0_mat))/float(len(class1_mat))
        rows = np.random.choice(class1_mat.index.values, 
                prop_sampled*class1_to_class0_ratio*len(class1_mat),
                                replace=False)
        class1_mat = class1_mat.ix[rows]
        print 'class1_len', len(class1_mat)
        rows = np.random.choice(class0_mat.index.values, 
                                prop_sampled*sample_limit, 
                                replace=False)
        class0_mat = class0_mat.ix[rows]
        print 'class0_len', len(class0_mat)
        eq_prior_sampled = eq_prior_sampled.append(class1_mat)
        eq_prior_sampled = eq_prior_sampled.append(class0_mat)
        del class0_mat, class1_mat
    #hdf['eq_prior_samp_all'] =  eq_prior_sampled
    eq_prior_sampled.to_hdf(HDF_file_name, 'data', format='t')
    
    del eq_prior_sampled
    #hdf.close()

eq_prior_sampler_from_hdf([3,1], .04, 'eq_prior_allfiles_.04')


def make_eq_prior_PCA_plots(df, folder):    
    """
    df = DataFrame <- HDF  from make_hdf
    """
    columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'y2']
    class1_mat = df[df['y2'] == 1]
    class0_mat = df[df['y2'] == 0]
    del df
    class1_proportion = float(len(class1_mat)/
                              (len(class0_mat)+len(class0_mat)))
    class0_proportion = 1 - class1_proportion
                          
    rows = np.random.choice(class1_mat.index.values, 
                            0.025*len(class1_mat),
                            replace=False)
    class1_mat = class1_mat.ix[rows]
    
    
    rows = np.random.choice(class0_mat.index.values, 
                            0.025*len(class0_mat), 
                            replace=False)
    class0_mat = class0_mat.ix[rows]
    
    pca = PCA(3, whiten=True)
    
    pca.fit(class1_mat[:][columns[0:5]])
    class1_mat_pca = pca.transform(class1_mat[:][columns[0:5]])
    class1_mat_pca = pd.DataFrame(class1_mat_pca)
    class1_mat_pca['y2'] = class1_mat['y2']
    del class1_mat
    
    pca.fit(class0_mat[:][columns[0:5]])
    class0_mat_pca = pca.transform(class0_mat[:][columns[0:5]])
    class0_mat_pca = pd.DataFrame(class0_mat_pca)
    del class0_mat
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(class1_mat_pca[0][:], 
               class1_mat_pca[1][:], 
               class1_mat_pca[2][:], c='r', marker='o')
    
    ax.scatter(class0_mat_pca[0][:], 
               class0_mat_pca[1][:], 
               class0_mat_pca[2][:], c='b', marker='^')
    plt.title("PCA file {}".format("all files"))
    plt.show()



