'''
Created on Oct 12, 2014

@author: asr13006
'''
import csv
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import os
import classification
import itertools

src = r'path'.replace('\\', '/')
os.chdir(src)

haberman = open('Haberman.csv', 'rb')

reader = csv.reader(haberman, delimiter=',')
df = pd.DataFrame()

i = 0
while True:
    
    try:
        df[i] = reader.next()
        i+=1
    except:
        break

df = df.transpose()
df = df.astype('float32')
X = np.asarray(df.iloc[:10,:3])
y = np.asarray(df.iloc[:10,3])
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.6, 
                                                    random_state=0)
##############################################################################
# PLOTTING THE X1, X2, X3 IN SCATTERPLOT
##############################################################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for c, i, m in [('r', 1, 'o'),('b', 2, '^')]:
    c = df[df.loc[:, 3] ==i]
    x1 = np.asarray(c.iloc[:,0])
    x2 = np.asarray(c.iloc[:,1])
    x3 = np.asarray(c.iloc[:,2])
    ax.scatter(x1, x2, x3, c=c, marker=m)

ax.set_xlabel('X1 Label')
ax.set_ylabel('X2 Label')
ax.set_zlabel('X3 Label')

plt.show()



##############################################################################
# kNN
##############################################################################
def combo_creator(feat_list):
    for n in range(0,len(feat_list)):
        for x in itertools.combinations(feat_list, n):
            if x != ():
                yield x

feat_list = [0, 1, 2, 3]
cls_col = 3
kNeighbors = 1

for feat_combo in combo_creator(feat_list): 
    
    
    
if __name__ == '__main__':
    pass
