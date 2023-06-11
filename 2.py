# -*- coding: utf-8 -*-
"""
Created on Tue May 30 22:59:04 2023

@author: Osman VARIŞLI 
"""

import pandas as pd

from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

data = pd.read_csv('diabetes.csv')

data['Glucose'].fillna(data['Glucose'].median(), inplace =True)

data['BloodPressure'].fillna(data['BloodPressure'].median(), inplace =True)

data['BMI'].fillna(data['BMI'].median(), inplace =True)

X=data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']]
y=data['Outcome']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)

models = []
models.append(('LinearRegression', LinearRegression(normalize=True)))
models.append(('Ridge', Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)))
models.append(('Lasso', Lasso(alpha=0.1,precompute=True,positive=True,selection='random',random_state=42)))
models.append(('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)))
models.append(('PolynomialFeatures', PolynomialFeatures(degree=2))) 
models.append(('SGDRegressor', SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000))) 

for name, model in models:
    if name=='PolynomialFeatures':
        #polynomial regression için özel bir durum var....
        X_train = model.fit_transform(X_train)
        X_test = model.transform(X_test)
        model = LinearRegression(normalize=True)
        
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    
    r2 = r2_score(y_test, predict)
    rmse = mean_squared_error(y_test, predict)
    mae=mean_absolute_error(y_test, predict)
    '''
    (r2),verilerin yerleştirilmiş regresyon hattına ne kadar yakın olduğunun istatistiksel bir ölçüsüdür
    (MAE), hataların mutlak değerinin ortalamasıdır
    (RMSE), karesi alınmış hataların ortalamasının kareköküdür:
    '''
    print(name , 'r2 : ', r2)
    print(name, 'rmse : ', rmse)
    print(name, 'mean_absolute_error : ', mae)
    print('-------------------------------------------')
    

    




