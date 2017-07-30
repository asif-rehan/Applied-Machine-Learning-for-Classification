'''
Created on Oct 15, 2014

@author: asif.rehan@engineer.uconn.edu
'''

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import itertools
from sklearn.metrics import accuracy_score
from random import randint
from sklearn.metrics.metrics import confusion_matrix
from sklearn.svm.classes import SVC
from sklearn.linear_model.perceptron import Perceptron
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.gaussian_process.gaussian_process import GaussianProcess
from sklearn.linear_model.base import LinearRegression


class feat_combo_classifier(object):
    
    def __init__(self, df):
        self.df = df
        

    def combo_creator(self, feat_list):
        for n in range(1, len(feat_list)+1):
            for x in itertools.combinations(feat_list, n):
                if x != ():
                    yield list(x)
    
    
    def feat_combo_classifier_func(self, cls_col,classifier, test_size, 
                              n_neighbors=1, sigma=1):
        """    
        df = PANDAS dataframe with columns for class and featuresS    
        """
        feat_list = list(self.df.columns)
        feat_list.remove(cls_col)
        print classifier
        y = np.asarray(self.df[cls_col].values)
        output = pd.DataFrame(columns=['FEATURES', 'TRAIN_ACCURACY', 
                                       'TEST_ACCURACY', 'CONFUSION_MATRIX'])
        for feat_combo in self.combo_creator(feat_list):  
            X = np.asarray(self.df[feat_combo].values)
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        test_size=test_size, 
                                        random_state=randint(0,100))
            if classifier == 'LDA':
                lin_dis = self.LDA_classifier(X_train, y_train, X_test, y_test)
                output = output.append([{'FEATURES':feat_combo,
                                'TRAIN_ACCURACY':round(lin_dis[0], 4)*100,
                                'TEST_ACCURACY': round(lin_dis[1], 4)*100,
                                'CONFUSION_MATRIX':lin_dis[2]}], 
                                ignore_index=True)
                
            
            elif classifier == 'QDA':
                quad = self.QDA_classifier(X_train, y_train, X_test, y_test)
                output = output.append([{'FEATURES':feat_combo,
                                    'TRAIN_ACCURACY':round(quad[0], 4)*100,
                                    'TEST_ACCURACY':round(quad[1], 4)*100,
                                    'CONFUSION_MATRIX':quad[2]}], 
                                    ignore_index=True)
            
            elif classifier == 'KNN':
                kneigh = self.KNN_classifier(X_train, y_train, X_test, 
                                        y_test, n_neighbors)
                output = output.append([{'FEATURES':feat_combo,
                                    'TRAIN_ACCURACY':round(kneigh[0], 4)*100,
                                    'TEST_ACCURACY':round(kneigh[1], 4)*100,
                                    'CONFUSION_MATRIX':kneigh[2]}], 
                                    ignore_index=True)
            
            elif classifier == 'PNN':
                pnn = self.PNN_classifier(X_train, y_train, 
                                          X_test, y_test, cls_col, sigma=sigma)
                output = output.append([{'FEATURES':feat_combo,
                                    'TRAIN_ACCURACY':'None',
                                    'TEST_ACCURACY':round(pnn[0], 4)*100,
                                    'CONFUSION_MATRIX':pnn[1]}], 
                                    ignore_index=True)
            
            elif classifier == 'SVM':
                sv = self.SVM_classifier(X_train, y_train, 
                                          X_test, y_test, cls_col)
                #print sv
                
                output = output.append([{'FEATURES':feat_combo,
                                    'TRAIN_ACCURACY':round(sv[0], 4)*100,
                                    'TEST_ACCURACY': round(sv[1], 4)*100,
                                    'CONFUSION_MATRIX':sv[2]}], 
                                    ignore_index=True)
            elif classifier == 'Perceptron':
                percep = self.Perceptron_classifier(X_train, y_train, 
                                          X_test, y_test, cls_col)
                
                output = output.append([{'FEATURES':feat_combo,
                                    'TRAIN_ACCURACY':round(percep[0], 4)*100,
                                    'TEST_ACCURACY': round(percep[1], 4)*100,
                                    'CONFUSION_MATRIX':percep[2]}], 
                                    ignore_index=True)
            elif classifier == 'Logistic_Regression':
                lr = self.Logistic_Regression(X_train, y_train, 
                                          X_test, y_test, cls_col)
                
                output = output.append([{'FEATURES':feat_combo,
                                    'TRAIN_ACCURACY':round(lr[0], 4)*100,
                                    'TEST_ACCURACY': round(lr[1], 4)*100,
                                    'CONFUSION_MATRIX':lr[2]}], 
                                    ignore_index=True)
            elif classifier == 'Decision_Tree':
                dtree = self.decision_tree_classifier(X_train, y_train, 
                                          X_test, y_test, cls_col)
                
                output = output.append([{'FEATURES':feat_combo,
                                    'TRAIN_ACCURACY':round(dtree[0], 4)*100,
                                    'TEST_ACCURACY': round(dtree[1], 4)*100,
                                    'CONFUSION_MATRIX':dtree[2]}], 
                                    ignore_index=True)
            elif classifier == 'ADABOOST':
                dtree = self.ADABOOST_classifier(X_train, y_train, 
                                          X_test, y_test, cls_col)
                
                output = output.append([{'FEATURES':feat_combo,
                                    'TRAIN_ACCURACY':round(dtree[0], 4)*100,
                                    'TEST_ACCURACY': round(dtree[1], 4)*100,
                                    'CONFUSION_MATRIX':dtree[2]}], 
                                    ignore_index=True)
            
            elif classifier == 'GP':
                dtree = self.ADABOOST_classifier(X_train, y_train, 
                                          X_test, y_test, cls_col)
                
                output = output.append([{'FEATURES':feat_combo,
                                    'TRAIN_ACCURACY':round(dtree[0], 4)*100,
                                    'TEST_ACCURACY': round(dtree[1], 4)*100,
                                    'CONFUSION_MATRIX':dtree[2]}], 
                                    ignore_index=True)
            elif classifier == 'Linear_Regression':
                linreg = self.ADABOOST_classifier(X_train, y_train, 
                                          X_test, y_test, cls_col)
                
                output = output.append([{'FEATURES':feat_combo,
                                    'TRAIN_ACCURACY':round(linreg[0], 4),
                                    'TEST_ACCURACY': round(linreg[1], 4),
                                    'CONFUSION_MATRIX':'--'}], 
                                    ignore_index=True)
            
        return output
        
            
    def LDA_classifier(self, X_train, y_train, X_test, y_test):
        lda = LDA()
        lda.fit(X_train, y_train)
        test_score = lda.score(X_test, y_test)
        train_score = lda.score(X_train, y_train)
        conf_mat = self.confusion_unravel(y_test, lda.predict(X_test))
        return train_score, test_score, conf_mat

    def QDA_classifier(self, X_train, y_train, X_test, y_test):
        qda = QDA()
        qda.fit(X_train, y_train)
        test_score = qda.score(X_test, y_test)
        train_score = qda.score(X_train, y_train)
        conf_mat = self.confusion_unravel(y_test, qda.predict(X_test))
        return train_score, test_score, conf_mat
    
    def KNN_classifier(self, X_train, y_train, X_test, y_test, n_neighbors):
        knn = KNeighborsClassifier(n_neighbors)
        knn.fit(X_train, y_train)
        test_score = knn.score(X_test, y_test)
        train_score = knn.score(X_train, y_train)
        conf_mat = self.confusion_unravel(y_test, knn.predict(X_test))
        return train_score, test_score, conf_mat
    
    def PNN_classifier(self, X_train, y_train, 
                       X_test, y_test, cls_col, sigma=1):
        """
        select neurons for each class
        """
        from scipy.stats import itemfreq
        classes = [item[0] for item in itemfreq(y_train)]
        weight_df = pd.DataFrame(preprocessing.normalize(X_train,norm='l2'))
        weight_df['y_train'] = y_train
        X_train_class_split = {}
        
        #makes weight matrices for each class
        for i in classes:
            X_train_class_split[i] = weight_df[weight_df['y_train'] == i]
            del X_train_class_split[i]['y_train']
            X_train_class_split[i] = np.asarray(X_train_class_split[i])
        #print X_train_class_split
        y_pred = []  
        input_X = preprocessing.normalize(X_test, norm ='l2')
        
        for x in input_X:
            discriminant = {}
            for c in classes:
                discriminant[c] = 0
                for k in X_train_class_split[c]:
                    discriminant[c] += np.exp((np.dot(k,x) - 1)/sigma**2)
            y_pred.append(classes[np.argmax(discriminant.values())])       
        conf = self.confusion_unravel(y_test, y_pred)
        return accuracy_score(y_test, y_pred, normalize=True), conf
    
    def SVM_classifier(self, X_train, y_train, X_test, y_test, cls_col):
        svc = SVC()
        svc.fit(X_train, y_train)
        test_score = svc.score(X_test, y_test)
        train_score = svc.score(X_train, y_train)
        conf_mat = self.confusion_unravel(y_test, svc.predict(X_test))
       
        return train_score, test_score, conf_mat

    def Perceptron_classifier(self, X_train, y_train, X_test, y_test, cls_col,
                              penalty=None, alpha=0.0001, eta0=1.0):
        mlp = Perceptron(penalty=penalty, alpha=alpha, fit_intercept=True,
                         eta0=eta0)
        mlp.fit(X_train, y_train)
        test_score = mlp.score(X_test, y_test)
        train_score = mlp.score(X_train, y_train)
        conf_mat = self.confusion_unravel(y_test, mlp.predict(X_test))
        return train_score, test_score, conf_mat
    
    def Logistic_Regression(self, X_train, y_train, X_test, 
                                       y_test, cls_col,
                              penalty='l2', tol=0.0001):
        log_reg = LogisticRegression(penalty=penalty, tol=tol, fit_intercept=True)
        log_reg.fit(X_train, y_train)
        test_score = log_reg.score(X_test, y_test)
        train_score = log_reg.score(X_train, y_train)
        conf_mat = self.confusion_unravel(y_test, log_reg.predict(X_test))
        return train_score, test_score, conf_mat
    
    def decision_tree_classifier(self, X_train, y_train, X_test, 
                                    y_test, cls_col):
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        test_score = dt.score(X_test, y_test)
        train_score = dt.score(X_train, y_train)
        conf_mat = self.confusion_unravel(y_test, dt.predict(X_test))
        return train_score, test_score, conf_mat
    
    def ADABOOST_classifier(self, X_train, y_train, X_test, 
                                    y_test, cls_col):
        clf = AdaBoostClassifier()
        clf.fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        train_score = clf.score(X_train, y_train)
        conf_mat = self.confusion_unravel(y_test, clf.predict(X_test))
        return train_score, test_score, conf_mat
    
    def GP_classifier(self, X_train, y_train, X_test, 
                                    y_test, cls_col):
        clf = GaussianProcess()
        clf.fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        train_score = clf.score(X_train, y_train)
        conf_mat = self.confusion_unravel(y_test, clf.predict(X_test))
        return train_score, test_score, conf_mat
    
    def linear_regression_classifier(self, X_train, y_train, X_test, 
                                    y_test, cls_col):
        clf = LinearRegression()
        clf.fit(X_train, y_train)
        test_Rsquared = clf.score(X_test, y_test)
        train_Rsquared = clf.score(X_train, y_train)
        return train_Rsquared, test_Rsquared
            
    def confusion_unravel(self, y_test, y_pred):
        cm = np.asarray(confusion_matrix(y_test, y_pred), dtype='float32')
        cm = np.around(cm/np.sum(cm)*100, 2)
        return [list(cm[i]) for i in range(len(cm))]
    
    
    