import pandas as pd
import numpy as np
from ISLP import load_data
from sklearn.pipeline import  make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression ,LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble  import AdaBoostRegressor,AdaBoostClassifier
from sklearn.metrics import r2_score ,accuracy_score ,mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

#L1 tends to shrink coefficients to zero whereas L2 tends to shrink coefficients evenly
#l1 weill helpfull for feature selection and overfitting
#l2 norm - eucliedan distance
#l2 will helpful for colinear/dependebcy checking and l2 reduces the variance
#providing penalty to all coefficents to make balanced solution

#l2 reglarization for the hiiers data

def ridge_regreesion():
    df = load_data('Hitters')
    np.random.seed(0)
    print(df.shape)
    print(df.dtypes)
    # df_array = df.to_numpy()
    # print(df['Salary'].isnull().sum())
    for column in df.columns:
        #check for the missing values
       if df[column].isnull().any():
           # for numerical value column fill mean of the column
           if df[column].dtype in ['int64','float']:
               mean_value = df[column].mean()
               df[column].fillna(mean_value,inplace=True)

               #for categorical values
           else:
               mod_value = df[column].mode()[0]
               df[column].fillna(mod_value,inplace=True)

    # print(df['Salary'].isnull().sum())
    #distrbution of univariant


    # sns.displot(df.Salary)
    # plt.xlabel('salary')
    # plt.ylabel('density')
    # plt.show()
    #
    # plt.figure(figsize=(15,10))
    #
    # dataplot = sns.heatmap(df.corr(),cmap="YlGnBu",annot=True,square=True)
    # plt.show()
    y = df['Salary']
    x = df.drop("Salary",axis=1)
    print(y)
    print(x['League'].value_counts())
    print(x['Division'].value_counts())
    print(x['NewLeague'].value_counts())



    x_array = x.to_numpy()
    y_array = y.to_numpy()
    # print(x_array)

    one_hot_columns = [13,14,18]
    encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    transformer = [('one hot encode',encoder,one_hot_columns)]
    ct = ColumnTransformer(transformers=transformer,remainder='passthrough')
    x_array = ct.fit_transform(x_array)
    print(x_array)


def hypotheis(theta,x):
    h_funct = np.dot(x,theta)
    return h_funct

def cost_function(h_funct,x,y):
    error = h_funct - y
    sse = np.sum(h_funct - y)**2 / 2
    return sse

def gradient_descent(x,y,alpha,l,n_iter):
    theta = np.zeros(x.shape[1])
    cost = []
    for i in range(n_iter):
        h_funct = hypotheis(theta,x)
        cost.append(cost_function(h_funct,x,y))
        error = np.dot(x.T,(h_funct - y))
        theta = theta - (alpha / 2 ) * np.sum(error) - (l /2 ) *  theta
        return theta,cost

theta,cost = gradient_descent(x,y,0.000001,0.001,1000)
print(theta)
