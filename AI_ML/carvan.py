#### caravan data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.pipeline import  make_pipeline
from sklearn.preprocessing import OneHotEncoder ,OrdinalEncoder ,StandardScaler
from sklearn.model_selection import  train_test_split
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn .model_selection import train_test_split,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,r2_score,mean_squared_error
from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder



##############################

def caravan_data():
    data = load_data('Caravan')
    print(data.shape)
    print(data.columns)
    print(data.dtypes)
    x = data.iloc[:,0:-1]
    print(x.shape)
    y = data.iloc[:,-1]
    print(y.shape)
    return x,y

#### heatmap is not work for these many features use pca

def pca_analysis(x,y):
    pca=PCA(n_components=35)
    scale = StandardScaler()
    scaled_data = scale.fit_transform(x)
    score = pca.fit_transform(scaled_data)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.cumsum())
    plt.scatter(score[:,0],score[:,1],score[:,2],score[:,3])
    plt.show()
    fig,axes = plt.subplots(1,2,figsize=(12,8))
    ax = axes[0]
    ticks = np.arange(pca.n_components_)+1
    ax.plot(ticks,pca.explained_variance_ratio_,marker="*",color="green")
    ax.set_xlabel("pca components")
    ax.set_ylabel("explained variance ratio")
    ax.set_title("explained ratio of pca component")
    ax = axes[1]
    ax.plot(ticks,pca.explained_variance_ratio_.cumsum(),marker="*",color="blue")
    ax.set_xlabel("pca components")
    ax.set_ylabel("cummulative sum of explained variance ratio")
    ax.set_title(" cummulative explained ratio of pca component")
    plt.legend()
    plt.show()
    label_encoder  = LabelEncoder()
    y_encoder = label_encoder.fit_transform(y)
    return y_encoder


from sklearn.cluster import  KMeans
def model_building(x,y_encoder):
    k = 10
    kmeans = KMeans(n_clusters=2, n_init=1, random_state=42)
    acc =[]
    fold = KFold(n_splits=k,random_state=42,shuffle=True)

    score_logistic = []
    score_random_forest = []
    score_xgboost = []
    score_kernel=[]
    model = make_pipeline(StandardScaler(),SVC(kernel="poly",degree=2)) #  using kernel svm   as model
    model_logistic = make_pipeline(StandardScaler(), LogisticRegression())
    model_random_forest = make_pipeline(StandardScaler(), RandomForestClassifier())
    model_xgboost = make_pipeline(StandardScaler(), XGBClassifier())
    for train_index, test_index in fold.split(x,y_encoder):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y_encoder[train_index], y_encoder[test_index]
        ### kernel model
    model.fit(x_train,y_train)
    y_p = model.predict(x_test)
    score = accuracy_score(y_test,y_p)
    score_kernel.append(score)

    # logistic model ###########################
    model_logistic.fit(x_train, y_train)
    y_p = model_logistic.predict(x_test)
    score = accuracy_score(y_test, y_p)
    score_logistic.append(score)
    ###random forest  ################################
    model_random_forest.fit(x_train, y_train)
    y_pred = model_random_forest.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    score_random_forest.append(accuracy)

    ##xg boost  ########################################

    model_xgboost.fit(x_train, y_train)
    y_p = model_xgboost.predict(x_test)
    score = accuracy_score(y_test, y_p)
    score_xgboost.append(score)

    #kmeans
    kmeans.fit(x_train, y_train)
    y_p = kmeans.predict(x_test)
    score = accuracy_score(y_test, y_p)
    acc.append(score)
    print("accuarcy of logistic is",np.mean(score_logistic))
    print("accuracy score random forest is",np.mean(score_random_forest))
    print("accuracy score of xgboost is",np.mean(score_xgboost))
    print("score of kernel is ",np.mean(score_kernel))
    print("accuracy of kmeans is",np.mean(acc))

def main():
    x,y =caravan_data()
    y_encoder = pca_analysis(x,y)
    model_building(x,y_encoder)
if __name__=="__main__":
    main()
