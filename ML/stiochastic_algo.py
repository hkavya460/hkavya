import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
intercept = df.insert(0,'intercept',1)
d = df.shape[1]
print(d)
x = df.iloc[:,0:d-2]
print(x)
y= df.iloc[:,d-2]


theta = np.zeros(6)
def hypothesis(x,theta):
    for i in range (0,d-2):
        h_funct = theta[i] *  x.iloc[:,i]
        return h_funct
def cost_function(h_funct,y):













