import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso ,Ridge

# L2 regularization / L2 norm / ridge  regression
#add penalty to loss function

df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
x = df.iloc[:,:-2]
y = df.iloc[:,-2]


def hypothesis(x,theta):
    h_funct = np.dot(x,theta)
    return h_funct
def error(x,h_funct,y):
    sse = 1/2 * sum((h_funct - y) ** 2)
    return sse
def gradient(x ,y,alpha,num_iteration,l):
    theta = np.zeros(x.shape[1])
    cost = []
    for i in range(num_iteration):
        h_funct = hypothesis(x, theta)
        sse = error(x,h_funct,y)
        cost.append(sse)
        cost_funct = np.dot(h_funct - y,x)
        # theta = theta - (alpha * cost_funct)    # without  l2 regularization
        theta = (theta - (alpha * l )/2) -(alpha /2  * cost_funct)   #with regularization
    return theta,cost

theta,cost= gradient(x,y,0.000001,1000,0.001)
# theta,cost = gradient(x,y,0.000001,1000,0.001)
print("theta values are:",theta)
print(cost)

# without regularization
def without_regularization():
    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    x = df.iloc[:,:-2]
    y = df.iloc[:,-2]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)
    model = LinearRegression()
    model.fit(x_train,y_train)
    print("coefficients are :",model.coef_)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    print("r2 score is :",r2)
without_regularization()

# l1 regularization / l1 norm / lasso regression

df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
x = df.iloc[:,:-2]
y = df.iloc[:,-2]
def hypothesis(x,theta):
    h_funct = np.dot(x,theta)
    return h_funct
def error(x,h_funct,y):
    sse = 1/2 * sum((h_funct - y) ** 2)
    return sse
def gradient(x ,y,alpha,num_iteration,l):
    theta = np.zeros(x.shape[1])
    cost = []
    for i in range(num_iteration):
        h_funct = hypothesis(x, theta)
        sse = error(x,h_funct,y)
        cost.append(sse)
        cost_funct = np.dot(h_funct - y,x)
        theta = theta* (1 - (alpha * l )) - (alpha * cost_funct)
    return theta ,cost

theta,cost = gradient(x,y,0.000001,1000,0.0001)
print("theta values are:",theta)
print("cost function values:",cost)


#using aclearn for lasso and ridge regression

df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
x = df.iloc[:,:-2]
y = df.iloc[:,-2]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)
model = Lasso(fit_intercept=True,alpha =0.00001)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("coefficients are :",model.coef_)
print(np.mean(y_test - y_pred) ** 2)
score =r2_score(y_test,y_pred)
print("r2 score of model is :",score)


#ridge  regularization

l2 = Ridge(fit_intercept=True,alpha=0.00001)
l2.fit(x_train,y_train)
y_preduction = l2.predict(x_test)
score =r2_score(y_test,y_preduction)
print("coefficent means theta values for ridge is",l2.coef_)
print(score)


