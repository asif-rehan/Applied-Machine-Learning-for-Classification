'''
Created on Sep 27, 2014

@author: asr13006
'''
import scipy.io
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas
from plot_lda_qda import *
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

cwd = 'folder_to_data'
os.chdir(cwd)
###############################################################################
#    Initial processing
###############################################################################
    
mat3 = scipy.io.loadmat('dataset3.mat')
mat2 = scipy.io.loadmat('Dataset2.mat')
mat1 = scipy.io.loadmat('Dataset1.mat')
            
f1 = np.hstack((np.hstack((mat3['f1'][0], mat2['f1'][0])), mat1['f1'][0]))
plt.hist(f1, 10, hold=True)
plt.show()
plt.xlabel('f1')
plt.ylabel('Freq')
plt.title(r'$\mathrm{Histogram\ of\ f1:}\ $')

scaled_f1 = np.divide(np.subtract(f1, np.mean(f1)), np.std(f1,ddof=1))  
plt.hist(scaled_f1, 10)
plt.show()
plt.xlabel('f1')
plt.ylabel('Density')
plt.title(r'$\mathrm{Histogram\ of\ scaled_f1:}\ $')

###############################################################################
#Step#1
###############################################################################
keys_list = ['f1', 'f2', 'f3', 'f4', 'f5','y2']
for keys in keys_list:
    print mat3[keys].size
##############################################################################
#X in numpy
###############################################################################

X = np.empty((1,1))
for i in xrange(mat3['f2'].size):
    x_i = []
    for keys in keys_list:
        x_i.append(mat3[keys][0][i])  
    if i==0:
        X = np.array([x_i])
    elif i>0:
        X = np.append(X, [x_i], axis=0)

##############################################################################        

#create pandas dataframe excluding rows with excess f5 values

df = pandas.DataFrame()
for key in keys_list:
    df[key] = mat3[key][0][:]
del mat3
#df_1 = df[df['y2'] > 0]
#df_0 = df[df['y2'] == 0]


X = np.asarray(df.iloc[:, 0:5].values)
y = np.ravel(df.iloc[:, 5:6].values)
X.head(3)
type(X)
y.head(3)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.40, 
                                                    random_state=0)
X_train.shape; X_test.shape
y_train.shape
lda = LDA()
y_pred = lda.fit(X_train, y_train, store_covariance=True).predict(X_test)
lda.score(X_test, y_test)
cm_lda = confusion_matrix(y_test, y_pred)
print cm_lda
plt.matshow(cm_lda)
plt.title('Confusion matrix for LDA')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
splot = plot_data(lda, X_test, y_test, y_pred, fig_index=2 * 0 + 1)
plot_lda_cov(lda, splot)
plt.axis('tight')





# QDA
qda = QDA()
y_pred = qda.fit(X_train, y_train, store_covariances=True).predict(X_test)
qda.score(X_test, y_test)
cm_qda = confusion_matrix(y_test, y_pred)
print cm_qda
plt.matshow(cm_qda)
plt.title('Confusion matrix for QDA')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
splot = plot_data(qda, X_test, y_test, y_pred, fig_index=2 * 0 + 2)
plot_qda_cov(qda, splot)
plt.axis('tight')
plt.suptitle('LDA vs QDA')
plt.show()







if __name__ == '__main__':
    pass
