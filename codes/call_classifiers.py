'''
Created on Nov 28, 2014

@author: asr13006
'''
from classification import feat_combo_classifier
import os
import sys
from pandas.io.pytables import read_hdf
import time
import numpy as np
from sklearn.externals.six import StringIO
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.tree.tree import DecisionTreeClassifier

def classify_it(df, classifier_name, filename, test_size=0.5, n_neighbors=1):
    sys.stdout = open('{0}_{1}.txt'.format(time.strftime(
                            '%Y-%m-%dT%H.%M.%S',time.localtime()), 
                                           filename), 'w')
    clssfr = feat_combo_classifier(df)
    output = clssfr.feat_combo_classifier_func('y2', classifier_name, 
                                      test_size=test_size, sigma=1, 
                                      n_neighbors=n_neighbors)
    print output
    sys.stdout.close()

cwd = r'path_to_data_file'.replace("\\", '/')
os.chdir(cwd)
data = read_hdf('DO_NOT_CHANGE_minmax_scaled.h5', 'data')

"""
###############################################################################
# LDA classification
###############################################################################
classify_it(data, 'LDA', 'LDA_Dataset1_sampled_eq_prior_testing')

###############################################################################
# QDA classification
###############################################################################
classify_it(data, 'QDA', 'QDA_Dataset1_sampled_eq_prior_testing')

###############################################################################
# QDA classification
###############################################################################
for k in [1,3,5]:
    classify_it(data, 'KNN', n_neighbors=k,
                'KNN_k={}_Dataset1_sampled_eq_prior_testing'.format(k))

"""
"""
###############################################################################
# PNN classification
###############################################################################
rows = np.random.choice(data.index.values, 0.05*len(data), replace=False)
pnn_sampled = data.ix[rows]
classify_it(pnn_sampled, 'PNN', 
            'PNN_Dataset1_sampled_eq_prior_testing',
            test_size = 0.2)

"""
###############################################################################
# SVM classification
###############################################################################
### testing on smaller sample
#rows = np.random.choice(data.index.values, 0.5*len(data), replace=False)
#svm_sampled = data.ix[rows]
#classify_it(svm_sampled, 'SVM', 'SVM_0.5_Dataset1_sampled_eq_prior_testing')
### on full sample
#classify_it(data, 'SVM', 'SVM_Dataset1_sampled_eq_prior_testing')

###############################################################################
# Perceptron classification
###############################################################################
#classify_it(data, 'Perceptron', 'SVM_Dataset1_sampled_eq_prior_testing')

###############################################################################
# Logistic Regression
###############################################################################
#classify_it(data, 'Logistic_Regression', 'Logistic_Reg_Dataset1_sampled_eq_prior_testing')
'''
###############################################################################
# Decision Tree
###############################################################################
classify_it(data, 'Decision_Tree', 'Decision_Tree_Dataset1_sampled_eq_prior_testing')
for decision tree visualization
columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'y1']
X = np.asarray(data[columns[0:5]].values)
y = np.asarray(data['y2'].values)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        test_size=0.5, 
                                        random_state=100)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
dot_data = StringIO()
with open("Decision_Tree_Vis_Dataset1_sampled_eq_prior_testing.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
tree.export_graphviz(clf, out_file=dot_data) 
os.unlink('Decision_Tree_Vis_Dataset1_sampled_eq_prior_testing.dot')
'''
###############################################################################
# AdaBoost
###############################################################################
classify_it(data, 'ADABOOST', 'ADABOOST_on DT_Dataset1_sampled_eq_prior_testing')

###############################################################################
# Gaussian Processes
###############################################################################
classify_it(data, 'GP', 'Gaussian_Processes_Dataset1_sampled_eq_prior_testing')
