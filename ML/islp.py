import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns



brain_cancer = load_data('BrainCancer')
print(brain_cancer.shape)
print(brain_cancer.columns)
print(brain_cancer.dtypes)
print(brain_cancer['diagnosis'].value_counts())
# print(brain_cancer['diagnosis'].isnull())
print(brain_cancer['loc'].value_counts())
v = brain_cancer['diagnosis']
print(v)
# one null value in the diagnosis feature
#finding the missing value
column_with_missingvalues = 'diagnosis'
data_indices = brain_cancer[brain_cancer[column_with_missingvalues].isnull()].index
print(data_indices)
#index 13 has no value in that
#assign unique value to the missing value
unique_categories =  brain_cancer[column_with_missingvalues].unique()

#by randomly assigning the unique value to the missing value
brain_cancer.loc[data_indices,column_with_missingvalues] = np.random.choice(unique_categories,len(data_indices))
print(brain_cancer['diagnosis'].isnull().sum())


#dataframe to array
df_array = brain_cancer.to_numpy()
print(df_array)

one_hot_columns = [0,1,2,5]
encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
transformer = [('one hot encoding',encoder,one_hot_columns)]
ct = ColumnTransformer(transformers =transformer,remainder='passthrough')
brain_cancer_encoded = ct.fit_transform(df_array)
print(brain_cancer_encoded)
df_brain_cancer_encoded = pd.DataFrame(brain_cancer_encoded,columns=ct.get_feature_names_out())
print(df_brain_cancer_encoded)




df = load_data('Hitters')
print(df['Salary'].isnull().sum())
for column in df.columns:
    if df[column].isnull().any():
        if df[column].dtypes in ['int64','float64']:
            mean_value = df[column].mean()
            df[column].fillna(mean_value,inplace=True)
        else:
            mod_value = df[column].mod()[0]
            df[column].fillna(mod_value,inplace=True)
print(df['Salary'].isnull().sum())



from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

x,y = make_classification(50,n_features=5,n_classes=2)

def kfold_cv(x,k):
    fold_size = len(x) // k
    folds =[]
    index = np.arange(len(x))
    for i in range (k):
        test_index = index[i * fold_size:(i+1) * fold_size]
        train_index = np.concatenate([index[:i * fold_size],index[(i+1) * fold_size:]])
        folds.append((train_index,test_index))
    return folds

folds = kfold_cv(x,10)
score =[]
for train_index,test_index in folds:
    x_train,x_test = x[train_index],x[test_index]
    y_train,y_test = y[train_index],y[test_index]
    model = LogisticRegression()
    model.fit(x_train,y_train)
    y_p = model.predict(x_test)
    acc = accuracy_score(y_test,y_p)
    score.append(acc)
print(np.mean(score))