import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#Loading the data (supervised linear regression data) 
df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
#print(df)
print(df.shape)
#x they all are features(Independent varibales) y is dependent variables 
x = df[["age","BMI","BP","blood_sugar","Gender"]]
y = df["disease_score_fluct"]

# Hypothesis Function  = theta.Transpose * X

def hypothesis(x,theta):
    h_funct= np.dot(x,theta)
    return h_funct
#Cost function is nothing but sum of square of errors  square of difference between the actual y - predicted y 
#our aim is to reduce the cost function the difference or erorr should be low 
def cost_function(h_funct,y):
    error = h_funct - y
    m = len(y)
    sse = np.sum(error **2) / (2 *m)
    return sse
    #Gradient descent is parameter update, update the theta untill it leads to global minima (reduce the cost function)
def gradient_descent(x,y,alpha,num_itera):
    theta = np.zeros(x.shape[1])
    cost =[]
    for i in range (num_itera):
        h_funct = hypothesis(x, theta)
        cost.append(cost_function(h_funct, y))
        error = h_funct - y
        gradient = np.dot(x.T,error)
        theta = theta - (alpha * gradient)
    return theta, cost




theta, cost= gradient_descent(x, y, 0.000001, 1000)
# theta ,cost= gradient_descent(x,y,0.0001,1000)
print("theta values are ",theta)
print("cost values ", cost)








