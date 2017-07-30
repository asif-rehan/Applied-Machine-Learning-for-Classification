'''
Created on Oct 19, 2014

@author: asr13006
'''
import os
import time
import sys
import scipy.io
import pandas as pd
from classification import feat_combo_classifier
from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt



def for_dataset1():
    cwd = r'path_to_datafile'.replace("\\", '/')
    os.chdir(cwd)
    
    mat3 = scipy.io.loadmat('dataset3.mat')
    mat2 = scipy.io.loadmat('dataset2.mat')
    mat1 = scipy.io.loadmat('dataset1.mat')
    
    columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'y2']
    cls_col = 'y2'
    n_neighbors = 1
    
    sys.stdout = open('{0}_{1}.txt'.format(time.strftime(
                            '%Y-%m-%dT%H.%M.%S',time.localtime()), 
                                           'LDA_Dataset1_file3_testing'), 'w')
    
    df = pd.DataFrame()
    for i in columns:
        df[i] = mat3[i][0][:]
    del mat3
    
    
    
    output = feat_combo_classifier(df, 'y2', 'LDA', test_size=0.5)
    print output
    sys.stdout.close()
    
def for_Haberman(classifier, n_neighbor=1, 
                 test_size = 0.5, pnn_sigma=1):
    cwd = r'path_to_datafile'.replace("\\", '/')
    os.chdir(cwd)
    Haberman = pd.read_csv('Haberman.csv')
    cls_col = 'y'
    
    #sys.stdout = open('{0}_{1}.txt'.format(time.strftime(
    #                        '%Y-%m-%dT%H.%M.%S',time.localtime()), 
    #                                       (classifier+'Haberman')), 'w')
    classify_Haberman = feat_combo_classifier(Haberman)
    output = classify_Haberman.feat_combo_classifier_func(cls_col, classifier, 
                                            test_size, n_neighbor, pnn_sigma)
    print output

def for_iris(classifier, n_neighbor=1, 
                 test_size = 0.5, pnn_sigma=1):
    cwd = r'path_to_datafile'.replace("\\", '/')
    os.chdir(cwd)
    iris = pd.read_csv('Iris.data', names=('Sl', 'sw', 'pl', 'pw', 'class'))
    iris = iris.dropna()
    cls_col = 'class'
    
    sys.stdout = open('{0}_{1}.txt'.format(time.strftime(
                            '%Y-%m-%dT%H.%M.%S',time.localtime()), 
                                           (classifier+'iris')), 'w')
    classify_Haberman = feat_combo_classifier(iris)
    output = classify_Haberman.feat_combo_classifier_func(cls_col, classifier, 
                                            test_size, n_neighbor, pnn_sigma)
    print output
    

    
#for_Haberman('PNN')
#for_iris('PNN')
