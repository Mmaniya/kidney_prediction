import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import median

dataset = pd.read_csv('ckd-dataset.csv')
X = dataset.iloc[:, 0:24].values
y = dataset.iloc[:, 24].values

def is_float(input):
  try:
    num = float(input)
  except ValueError:
    return False
  return True

for i in range(0,399):
    if y[i] == 'ckd':
        y[i] = 1
    else:
        y[i] = 0
y = y.astype(int)

for a in range(0, 399):
    if X[a][5] == 'normal':
        X[a][5] = 0
    if X[a][5] == 'abnormal':
        X[a][5] = 1
        
for a in range(0, 399):
    if X[a][6] == 'normal':
        X[a][6] = 0
    if X[a][6] == 'abnormal':
        X[a][6] = 1
        
for a in range(0, 399):
    if X[a][7] == 'notpresent':
        X[a][7] = 0
    if X[a][7] == 'present':
        X[a][7] = 1
        
for a in range(0, 399):
    if X[a][8] == 'notpresent':
        X[a][8] = 0
    if X[a][8] == 'present':
        X[a][8] = 1
        
for a in range(0, 399):
    for b in range(18, 24):
        if X[a][b] == 'yes' or X[a][b] == 'good':
            X[a][b] = 0
        if X[a][b] == 'no' or X[a][b] == 'poor':
            X[a][b] = 1
    
for a in range(0,399):
    for b in range(0, 24):
        if(isinstance(X[a][b], int)):
            X[a][b] = float(X[a][b])
        elif(isinstance(X[a][b], str)):
            if(is_float(X[a][b])):
                X[a][b] = float(X[a][b])
                
totals = [0] * 24
added = [0] * 24           
for a in range(0, 399):
    for b in range(0, 24):
        if(isinstance(X[a][b], float)):
            totals[b] += X[a][b]
            added[b] += 1
            
averages = [0] * 24          
for a in range(0, 24):
    averages[a] = totals[a] / added[a]
 
c = 0
for a in range(0, 399):
    for b in range(0, 24):
        if(isinstance(X[a][b], float) == 0):
            X[a][b] = averages[b]
            c += 1
    
X = X.astype(float)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Creating the model with just 2 features

"""import seaborn as sns
from sklearn import datasets
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 100 == 0):
                print(f'loss: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()
model = LogisticRegression(lr=0.1, num_iter=1000000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
model.theta
from sklearn.metrics import f1_score
f1_score(y_test, preds, average='binary')
from sklearn.metrics import accuracy_score
accuracy_score(y_test, preds)"""


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
r accuracy_score(y_test, y_pred)

from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='binary')
