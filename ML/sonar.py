import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import  random
#load the data
def logistic_simulation():
    df = pd.read_csv("binary_logistic_regression_data_simulated_for_ML.csv")
    x = df[["gender","age","blood_pressure","LDL_cholesterol"]]
    y = df["disease_status"]

    # split the data in to test and train using sklearn test_train
    x_train ,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)
    # after splitting the data train the data to fit
    model = LogisticRegression()
    model.fit(x_train,y_train)
    print(model)
    #test the model with tested data
    y_pred = model.predict(x_test)

    score= accuracy_score(y_test,y_pred)
    print("accuracy of the model from simulated data is:",score)

#sonar data
def logistic_sonar():
    np.random.seed(42)
    data1= pd.read_csv("sonar data.csv")
    x = data1.iloc[:,:-1]
    y = data1.iloc[:,-1]
    print(x,y)
# split the data using train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)
#fit the train data
    model = LogisticRegression()
    model.fit(x_train,y_train)
    y_predct =model.predict(x_test)
    accuacyscore = accuracy_score(y_test,y_predct)
    print("accuracy score of sonar is :",accuacyscore)

#k fold cross validation for sonar data using sklearn
def k_fold_sonar():
    np.random.seed(42)
    df = pd.read_csv("sonar data.csv")
    print(df.shape)
    x = df.iloc[:,:-1]
    print(x.shape)
    y = df.iloc[:,-1]
    print(y.shape)
    k = 10
    model = LogisticRegression()
    acc_score = []
    fold = KFold(n_splits=k,random_state=None,shuffle=True)
    for train_index ,test_index  in fold.split(x):
        x_train,x_test = x.iloc[train_index],x.iloc[test_index]
        y_train,y_test = y.iloc[train_index],y.iloc[test_index]
# fitting the training model  for each fold
        model.fit(x_train,y_train)
# fitting for testing model
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_pred,y_test)
        acc_score.append(acc)
    #mean of accuarcy score
    mean_accuarcy_acore = sum(acc_score) / k
    print("accuracy score of each fold is:",acc_score)
    print("mean accuracy score of the fold is :",mean_accuarcy_acore)




# k_fold_sonar()
#logistic_sonar()
#logistic_simulation()
def data_normalize():
    np.random.seed(42)
    data1 = pd.read_csv("sonar data.csv")
    x = data1.iloc[:,:-1]
    y = data1.iloc[:,-1]
    for i in range (x.shape[1]):
        max_value = x.iloc[:,i].max()
        min_value = x.iloc[:,i].min()
        x.iloc[:,i] = (x.iloc[:,i] - min_value) / max_value - min_value
    #print("Normalized data:")
    #print(x,y)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)
    model = LogisticRegression()
    model.fit(x_train,y_train)
    y_predct = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_predct)
    return accuracy

    # k = 10
    # accuraccy = []
    # k_fold = KFold(n_splits=k,random_state=None,shuffle=True)
    # classifier = LogisticRegression()
    # for train_index,test_index in k_fold.split(x):
    #     x_train,x_test = x.iloc[train_index],x.iloc[test_index]
    #     y_train,y_test = y.iloc[train_index],y.iloc[test_index]
    #
    #     classifier.fit(x_train,y_train)
    #     y_pred1 = classifier.predict(x_test)
    #     acc2 = accuracy_score(y_pred1,y_test)
    #     accuraccy.append(acc2)
    # return accuraccy


# accuraccy = data_normalize()
# print("accuracy  score after normalization",accuraccy)

data1 = pd.read_csv("sonar data.csv")
x = data1.iloc[:, : -1]
y = data1.iloc[:, -1]
def standardization(data1,x,y):

    for i in range (x.shape[1]):
            mean_value = x.iloc[:,i].mean()
            standard_devi = x.iloc[:,i].std()
            x.iloc[:,i] = (x.iloc[:,i] - mean_value )/ standard_devi
    print("standardization :")
    print(x)
    return x , y

#x,y =standardization(data1,x,y)
# print(x)

#with sklearn

def std_scaler():
    data1 = pd.read_csv("sonar data.csv")
    x = data1.iloc[:, : -1]
    y = data1.iloc[:, -1]
#divide the data for  training and testing
    print(data1.shape)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)
    print(x_train.shape)
#split the trainig data into training and validation
    x_train_temp ,x_val,y_train_temp,y_val = train_test_split(x_train,y_train,test_size=0.20)
    print(x_train_temp)
    print(x_val.shape)
    object = StandardScaler()
    model = object.fit(x_train_temp,y_train_temp)
    object.transform(x_val)
    y_pred = model.transform(x_test)

    mse = mean_squared_error(y_test,y_pred)
    print("mean square error is",mse)


std_scaler()
