# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 02:40:33 2022

@author: Yibo Wang
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np


############## hyy.csv ##############

data_frame = pd.read_csv("Data_update.csv")                                     # File name
X = np.array(data_frame)[:, 4:17]                                               # Variable E to Q
y = np.array(data_frame)[:, 0:2]                                                # ID and results

mislabled = 1000
model = None

for echo in range(100):
    
    print("Number of iteration: %d" % (echo+1))
                                                                                
    X_train, X_test, y_train1, y_test1 = train_test_split(X, y, test_size=0.3, random_state=np.random)
    y_train = y_train1[:, 1]                                                    # Only keep the results in y
    y_test = y_test1[:, 1]
    
    gnb = GaussianNB()                                                          # The Gaussian Naive Bayes framework
    y_pred = gnb.fit(X_train, y_train).predict(X_test)                          # The predict results on X_test
    
    if (y_test != y_pred).sum() < mislabled:
        
        mislabled = (y_test != y_pred).sum()
        model = gnb.fit(X_train, y_train)
        print("Number of mislabeled points out of a total %d points : %d"
              % (X_test.shape[0], (y_test != y_pred).sum()))
    
final_pred = model.predict(X)
pd.DataFrame(model.predict_proba(X)).to_csv('prob.csv')
print("Number of mislabeled points out of a total %d points : %d"
      % (X.shape[0], (y[:, 1] != final_pred).sum()))

# print(np.sum(y[:, 1] == 1))
# print(np.sum(y[:, 1] == 2))
# print(np.sum(y[:, 1] == 3))
# y_pred_1 = 0
# y_pred_2 = 0
# y_pred_3 = 0

# for i in range(len(final_pred)):
#     if y[i, 1] != final_pred[i]:
            
#         if y[i, 1] == 1:
#             print(i)
#             y_pred_1 += 1
#             # print(gnb.predict_proba(X_test)[i, 0]) 
#         if y[i, 1] == 2:
#             y_pred_2 += 1
#         if y[i, 1] == 3:
            
#             y_pred_3 += 1

# print(y_pred_1, y_pred_2, y_pred_3)
    