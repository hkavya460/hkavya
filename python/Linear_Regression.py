#Name = H kavya
#Objective = Machine learning
#Date = 11/1/2024

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from sympy import symbols , diff,lambdify

#build a model for the data  simulated_data_multiple_linear_regression_for_ML.csv
#in the above data taking disease_score as target to build model
#1)load the file
def load_file():
    file = pd.read_csv("C:\\Users\\User\\Downloads\\simulated_data_multiple_linear_regression_for_ML.csv")
    #print(file)
    x = file[["age","BMI","BP","blood_sugar","Gender"]]  # features of the data
   # print(x)
    y = file[["disease_score"]] # target of the data
# split the data for train the data ,splitted size is 70 :30
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)
    print(x_train)
    print(y_train)
    print(x_test.shape)
    model = linear_model.LinearRegression()
#fit the model using train data
    model.fit(x_train,y_train)
    print(model)
#predict model for test data
    y_predct = model.predict(x_test)
    print(y_predct)
#comparing the models by r2_score if the r2_score is between 0 to 1 if r2_score =1 model is fitt if it is 0 then the model is not good
    r2 = r2_score(y_test,y_predct)
    print("r2 score of disease score is ;",r2)
#target as disease_score_fluct
def dis_flut_scor():
    file = pd.read_csv("C:\\Users\\User\\Downloads\\simulated_data_multiple_linear_regression_for_ML.csv")
    # print(file)
    x = file[["age", "BMI", "BP", "blood_sugar", "Gender"]]  # features of the data
    y_label = file[["disease_score_fluct"]]
    x_train,x_test,y_train,y_test=train_test_split(x,y_label,test_size=0.30)
    print(x_train)
    print(y_train.head)
    model2 = linear_model.LinearRegression()
    model2.fit(x_train,y_train)
    y_predct2 = model2.predict(x_test)
    print(y_predct2)
    r2 = r2_score(y_test,y_predct2)
    print("r2 score of disease_fluct_score is :",r2)

#implement multplication of matrices A_transpose * A
def multiply_matrices(A):
    A_transpose = np.transpose(A)
    print(A_transpose)
    product  = np .dot (A_transpose,A)  # multiply A transpose with A
    print(product)

#implement the function h(x) = theta * x + theta1 * x1 for given theta and x values and plot the graph for the function
#range [-100,100,100]
def implement_funct():
    theta = 3
    x = 1
    theta1 = 2
    #h(x)= theta * x + theta1 * x1
    x = np.linspace(-100,100,100)
    y=2 *x +3
    plt.plot(x,y)
    plt.title("function h(x)=2 *x +3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
# implement h(x)= theta * x + theta1 * x1 + theta2 * x2^2
def implement_function_2():
    theta = 4
    theta1= 3
    theta2=2
    #y = 2x^2 + 3x +4
    x = np.linspace(-10,10,100)
    #print(x)
    y =2 *  x ** 2  + 3 * x + 4
    print(y)
    plt.plot(x,y)
    plt.title("function h(x)=2 *  x ** 2  + 3 * x + 4")
    plt.xlabel("x_values")
    plt.ylabel("y_values")
    plt.show()


#implement gaussian PDF mean = 0.5,sigma = 15 in the range [-100,100,100]
# f(x)=(1/√(2πσ2))*  (e[-(x-μ)^2]/2σ^2#gaussian probability density frequency distribution - which is symmetric about the mean
# for calculating  normal probability density frequency  distribution used norm.pdf from scipy
# x = np.linspace(-100,100,100)
# plt.plot(x,norm.pdf(x))
# plt.title("Normal PDF")
# plt.show()
def gaussian_pdf():
    mean = 0    # for given ean and standard devaitation using scipy norm
    std = 15
    x = np.linspace(-100,100,100)
    # y = norm(loc=mean,scale=std).pdf(x)
    # plt.plot(x,y,c="blue")
    # plt.title("Normal PDF")
    # plt.show()
#f(x)=(1/√(2πσ2))*  (e[-(x-μ)^2]/2σ^2
    y = 1 / (np.sqrt(2 * math.pi) * std) * np.exp(-1 / 2 * ((x - mean) / std) ** 2)
    plt.plot(x,y)
    plt.title("Normal PDF")
    plt.show()
#implement y = x ^2 ,and its derivative
def function():
    #y = x ** 2
    x = np.linspace(-100,100,100)
    y = x ** 2
    #plt.plot(x,y)
   # plt.title("f(x) = y ** 2 ")
    #plt.show()
#derivative of y i.e y (d/dx)
#for derivative using derive from sympy
#differentation y = 2x
    x = symbols('x')

    expr = x ** 2
    derivative = diff(expr,x )
    derivative_func =  lambdify(x, derivative, 'sympy')
    x_values = np.linspace(-100,100,100)
    y_values = derivative_func(x_values)
    plt.plot(x_values,y_values)
    plt.title("derivative of f(x)")
    plt.show()
def main():
    function()
    implement_funct()
    implement_function_2()
    gaussian_pdf()
    load_file()
    dis_flut_scor()

   A = np.array([[1,2,3],[4,5,6]])
   multiply_matrices(A)
if __name__=="__main__":
    main()

