import random

import pandas as pd
import numpy as np

df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
x =df[["age","BMI","BP","blood_sugar","Gender"]]
y =df["disease_score"]
theta = np.zeros(x.shape[1])

# def hypothesis(x,theta):
#     h_funct = np.dot(x,theta)
#     return h_funct
# def cost_function(h_funct,y):
#     m = len(y)
#     error = h_funct - y
#     sse = np.sum(error  ** 2)/ (2 * m)
#     return sse
# def gradient(x,y,alpha,num_iter):
#     m = len(y)
#     cost =[]
#     theta = np.zeros(x.shape[1])
#     for i in range (1000):
#         h_funct = hypothesis(x,theta)
#         cost.append(cost_function(h_funct,y))
#         error = h_funct - y
#         gradient = np.dot(x.T ,error)/m
#         theta = theta - (alpha * gradient)
#     return theta ,cost
# theta ,cost = gradient(x,y,0.0001,1000)
# print("the theta value ia :",theta)
# print("the cost value is:",cost)

# stochastic descent algorithm
def hypothesis(x,theta):
    h_funct = np.dot (x,theta)
    return h_funct
def cost_function(h_funct,y):

    error = h_funct - y
    sse = np.sum(error ** 2) / (2)
    return sse
def stochastic_grdaient(x,y,alpha,num_iter):
    cost = []
    theta = np.zeros(x.shape[1])
    random_index = np.random.randint(0, len(y))
    x_i = x.iloc[random_index]

    y_i = y.iloc[random_index]
    for i in range (num_iter):
        h_funct = hypothesis(x_i,theta)
        cost.append(cost_function(h_funct,y_i))
        error = h_funct - y_i
        gradient = np.dot (x_i.T,error)
        theta = theta - (alpha * gradient)
    return theta,cost


theta,cost = stochastic_grdaient(x,y,0.00001,1000)

print("cost is :", cost)
print("theta value is :",theta)




#logstic regression
#y = 1 - h_funct * x j

