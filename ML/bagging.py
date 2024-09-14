## ensemble models (Bagging and Decision Tree )

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold ,cross_val_score
from sklearn.ensemble import BaggingRegressor,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , mean_squared_error
from sklearn.tree import plot_tree
from matplotlib.pyplot import  plot


####################################################

# using Logistic regression data build the decision tree and bagging

df = pd.read_csv("binary_logistic_regression_data_simulated_for_ML.csv")
print(df)
x =df.iloc[:,:-1]
y= df.iloc[:,-1]
#splitting the data for trainig and testing

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
object = DecisionTreeClassifier()
dc = BaggingClassifier(object,n_estimators=12,random_state=22)
model = dc.fit(x_train,y_train)
print(model)
y_predct = dc.predict(x_test)
print(y_predct)
score = accuracy_score(y_test,y_predct)
print("accuracay score is :",score)

plt.Figure(figsize=(30,10))
plot_tree(dc.estimators_[0],feature_names=x.columns)
plt.show()
outcome = df.disease_status.value_counts()
print(outcome)