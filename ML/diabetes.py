import pandas as pd
import numpy as np
from ISLP import load_data
from sklearn.preprocessing import StandardScaler
import  matplotlib.pyplot as  plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


######################################################
diabet  =datasets.load_diabetes()

df = pd.DataFrame(diabet.data)
print(df.shape)
print(df.describe())

# corr_matrix = []
# corr_matrix = df.corr()
# sns.heatmap(corr_matrix,annot=True,square=True)
# plt.show()
# plt.boxplot(df)
# plt.show()

y = pd.DataFrame(diabet.target)
x =  df.iloc[:,:]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42,shuffle=True)

model = make_pipeline(StandardScaler(),LinearRegression())
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
score = r2_score(y_test,y_pred)
print(score)

scale = StandardScaler()

mod = LinearRegression()
sclaer_trans_train = scale.fit_transform(x_train,y_train)
scaler_trans_test = scale.fit_transform(x_test,y_test)


mod.fit(sclaer_trans_train,y_train)
y_pred = mod.predict(scaler_trans_test)
score = r2_score(y_test,y_pred)
print(score)

#xgboost
xg_model = XGBRegressor()
xg_model.fit(sclaer_trans_train,y_train)


modl = SVC(kernel="poly",C=10)
modl.fit(sclaer_trans_train,y_train)
y_p = modl.predict(scaler_trans_test)
sc = r2_score(y_test,y_p)
print(sc)




