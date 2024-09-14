#implement the sigmoid function
#g(z) = 1/1+ exp(- z**2)
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
x = df[["age","BMI","BP","blood_sugar","Gender"]]
y= df["disease_score"]

#logistic algorithm
#p(x)=  y(1-h_funct)* x
theta = np.zeros(x.shape[1])
h_funct = np.dot (x,theta)
print(h_funct)
for i in range ():
    p(x) =y *