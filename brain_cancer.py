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
import  warnings
#simple impute for missing values
##############################################################
def load_brain_data():
    np.random.seed(0)
    brain_cancer = load_data('BrainCancer')
    print(brain_cancer.shape)
    print(brain_cancer.columns)
    print(brain_cancer.dtypes)
    y = brain_cancer['status']
    x_brain_cancer  = brain_cancer[['sex', 'diagnosis', 'loc', 'ki', 'gtv', 'stereo','time']]
    y_array = y.to_numpy()
    return x_brain_cancer ,y_array

#in the dataframe the diagnosis has one null value replacing null value with random value


def prepocessing(x_brain_cancer):
    #finding missing and null values
    column_missing_value ='diagnosis'
    data_indices = x_brain_cancer[x_brain_cancer['diagnosis'].isnull()].index
    unique_categories = x_brain_cancer['diagnosis'].unique()
    x_brain_cancer.loc[data_indices,column_missing_value ] = np.random.choice(unique_categories, len(data_indices))
    print(x_brain_cancer['diagnosis'].isnull().sum())

    #converting dataframe to array
    x_brain_cancer_array = x_brain_cancer.to_numpy()
    #one hot encoding for categorical values 0,1, 2,5
    one_hot_coulmns = [0,1,2,5]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    transformers = [('one_hot_encoding', encoder, one_hot_coulmns)]
    ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
    x_brain_cancer_array_encoded = ct.fit_transform(x_brain_cancer_array)
    return x_brain_cancer_array_encoded
def pca_analysis(x_brain_cancer_array_encoded):
    pca = PCA(n_components=4)
    scaler = StandardScaler()
    scaled_brain_array_encoded = scaler.fit_transform(x_brain_cancer_array_encoded)
    score = pca.fit_transform(scaled_brain_array_encoded)

    print(pca.explained_variance_ratio_)
    plt.scatter(score[:,0],score[:,1],score[:,2],color="green")
    plt.xlabel("pca components")
    plt.title(" first 3 pca components of brain cancer data")
    fig,axes = plt.subplots(1,2,figsize=(12,6))
    ticks = np.arange(pca.n_components_)+1
    ax = axes[0]
    ax.set_xlabel("pca components")
    ax.set_ylabel("variance ratio")
    ax.set_title("pca component and explained variance ratio")
    ax.plot(ticks,pca.explained_variance_ratio_,marker="*",color="blue")
    ax = axes[1]
    ax.plot(ticks,pca.explained_variance_ratio_.cumsum(),marker="+",color="red" )
    ax.set_xlabel("pca components")
    ax.set_ylabel(" cummulative sum of variance ratio")
    ax.set_title("cummulative curve")
    plt.show()

# kfold cross validation
def kfold_crossvalidation(x_brain_cancer_array_encoded,y_array):
    k = 10
    score_logistic  =[]
    score_random_forest =[]
    score_xgboost =[]
    model_logistic  = make_pipeline(StandardScaler(), LogisticRegression())
    model_random_forest = make_pipeline(StandardScaler(), RandomForestClassifier())
    model_xgboost = make_pipeline(StandardScaler(), XGBClassifier())
    fold = KFold(n_splits=k,random_state=42,shuffle=True)
    for train_index,test_index in fold.split(x_brain_cancer_array_encoded,y_array):
        x_train,x_test = x_brain_cancer_array_encoded[train_index],x_brain_cancer_array_encoded[test_index]
        y_train,y_test = y_array[train_index],y_array[test_index]

        ###logistic regression
        model_logistic.fit(x_train,y_train)
        y_p = model_logistic.predict(x_test)
        score= accuracy_score(y_test,y_p)
        score_logistic.append(score)
        ###random forest  ################################
        model_random_forest.fit(x_train,y_train)
        y_pred = model_random_forest.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        score_random_forest.append(accuracy)

        ##xg boost

        model_xgboost.fit(x_train, y_train)
        y_p = model_xgboost.predict(x_test)
        score = accuracy_score(y_test, y_p)
        score_xgboost.append(score)
    return score_logistic,score_random_forest,score_xgboost



def main():
    x_brain_cancer,y_array = load_brain_data()
    x_brain_cancer_array_encoded =prepocessing(x_brain_cancer)
    pca_analysis(x_brain_cancer_array_encoded)
    k = 10
    score_logistic,score_random_forest,score_xgboost = kfold_crossvalidation(x_brain_cancer_array_encoded,y_array)
    avg_logistic = np.sum(score_logistic) / k
    avg_random_forest = np.sum(score_random_forest) / k
    avg_xgboost = np.sum(score_xgboost)/ k
    print("average of logistic regression is ",avg_logistic)
    print("average of random forest  is ",avg_random_forest)
    print("average of xgboost is ", avg_xgboost)


if __name__=="__main__":
    main()

########################################################################




##################################################################################

