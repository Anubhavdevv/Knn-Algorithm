#Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import warnings 
warnings.filterwarnings("ignore")  #Ignore the Warning 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import metrics as m
from sklearn.metrics import classification_report


#Importing Dataset
df = pd.read_csv('train.csv')
df.shape

df.head()

X = df.drop(['label'],axis = 1)
Y = df['label']

train_x,test_x,train_y,test_y= train_test_split(X, Y, test_size= 0.3, random_state=20)  

#data = df.values
#data

def elbow_curve(k):
    empty_lst = []   #empty list

    for i in k:   #instance for knn
        clf = knn(n_neighbors=i)
        clf.fit(train_x,train_y)
        tmp = clf.predict(test_x)
        tmp = m.accuracy_score(tmp,test_y)
        error = 1-tmp
        empty_lst.append(error)
   
    return empty_lst
    
k = range(1,10)    

test = elbow_curve(k)

plt.plot(k, test)
plt.xlabel('K Neighbors')
plt.ylabel('Test error')
plt.title('Testing elbow_curve')
plt.show()

emp={}
for i in range(1,10):
    emp[i]=np.interp(i,k,test)

val=1
for j in range(1,10):
    if(val>emp[j]):
        val=emp[j]
        num=j
va = knn(n_neighbors=num)
va.fit(train_x,train_y)

pred = va.predict(test_x)

print(classification_report(test_y, pred)) 
