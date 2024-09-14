import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier ,AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder ,OrdinalEncoder ,LabelEncoder
from sklearn.compose import ColumnTransformer


# use simpleimpute for  missing values
##########################################################
#Using the breast cancer data (Logistic regression)

def breast_cancer_logistic():
    #loading the data and converting into numpy array
    df = pd.read_csv("b_cancer.csv")
    df_numpy = df.to_numpy()
    x = df_numpy[:,0:9]
    y = df_numpy[:,-1]
    return x,y


#the data of the below columns which are they have cateogrical value so converting using preprocessing library
# preprecessing ordinal encoding and one hot encoding

def preprocesing(x,y):
    ordinal_cols= [0,2,3,5]
    onehot_columns = [8, 7, 6, 4, 1]

    t = ColumnTransformer(transformers=[
        ('one_hot_encoding',OneHotEncoder(sparse_output=False),onehot_columns),
        ("ordinal_encoding",OrdinalEncoder(),ordinal_cols)
    ],remainder='passthrough')
    x = t.fit_transform(x)


    #splitting the data into 70% for training and remianing will be testing
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42,shuffle=True)

#Label encoder to normalize the labels
    label = LabelEncoder()

#fitted the data
    label.fit(y_train)
    y_train = label.transform(y_train)
    y_test = label.transform(y_test)
    model = RandomForestClassifier()
    model.fit(x_train,y_train)
    #predict the model by passing testing data
    y_pred = model.predict(x_test)

    #computing the accuracy score
    acc = accuracy_score(y_test,y_pred)
    print("accuracy score is ",acc)

    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(x_test)

    # Visualize SHAP summary plot
    shap.summary_plot(shap_values, x_test)
    shap.summary_plot(shap_values[1],x_test)

def main():
    x,y = breast_cancer_logistic()
    preprocesing(x,y)


main()


