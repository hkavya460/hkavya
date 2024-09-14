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

#####Hitters data


def load_hitters_data():
    df = load_data('Hitters')
    np.random.seed(0)
    print(df.shape)
    print(df.dtypes)

    return df
def preprocessing(df):
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
    # plt.figure(figsize=(15,10))
    # dataplot = sns.heatmap(df.corr(),cmap="YlGnBu",annot=True,square=True)  #correlation map
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
    return x_array ,y_array
# model based feature selection using scikit learn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def model_building(x_array,y_array):

    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.30, random_state=42, shuffle=True)
    model = make_pipeline(StandardScaler(), AdaBoostRegressor())
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = r2_score(y_test, y_pred)
    print(score)

    # ridge = 0.4834
    # Adabosst = 0.46311
    model_xg = make_pipeline(StandardScaler(), XGBRegressor())
    model_xg.fit(x_train, y_train)
    y_p = model_xg.predict(x_test)
    score = r2_score(y_test, y_p)
    print("xg scoreis", score)
    print("mean square error is ",np.sqrt(mean_squared_error(y_test, y_p)))
    print("mean square error is ",np.sqrt(mean_squared_error(y_test, y_pred)))

def main():
    df =load_hitters_data()
    x_array,y_array =preprocessing(df)
    model_building(x_array,y_array)

if __name__=="__main__":
    main()


