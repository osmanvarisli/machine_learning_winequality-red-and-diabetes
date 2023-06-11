# -*- coding: utf-8 -*-
"""
Created on Tue May 30 21:28:18 2023

@author: Osman VARIŞLI
"""

import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("winequality-red.csv")

q = []
for i in df['quality']:
    if i in (3, 4, 5):
        q.append('1')
    elif i in (6,7,8):
        q.append('2')

df['quality'] = q



x=df.drop('quality', axis=1)
y=df['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 13)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


models = []
models.append(('NBayes', GaussianNB()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('Random Forest',RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))

plt.figure(figsize=(8,6), dpi=100)
sns.set(font_scale = 1.1)


for name, model in models:

    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    
    acc_score = accuracy_score(y_test, predict)
    print(name," Başarı Oranı:", acc_score*100)
    conf_matrix=confusion_matrix(y_test, predict)
    
    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )
    ax.set_xlabel("Tahmin Değerler", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    ax.set_ylabel("Gerçek Değer", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    ax.set_title(name+" modeli için", fontsize=14, pad=20)
    plt.show()
    #print(conf_matrix)


 




 