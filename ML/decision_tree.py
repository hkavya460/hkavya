from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree



#steps to buold decision tree
#data preprocessing
#fitting decision tree in to training set
#test accuaracy of the result
#visualize the test set result
#decision tree for linear regression
##########################################
df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
print(df)
print(df.shape)
x =df.iloc[:,:-2]
y =df.iloc[:,-2]
print(x.shape)
print(y.shape)
dtree = DecisionTreeRegressor()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)
dtree.fit(x_train,y_train)
# predciting for testing
y_pred = dtree.predict(x_test)
score = r2_score(y_test,y_pred)
print("score of the linear regression is:", score)
# dtree.fit(x,y)

plt.figure(figsize=(30,20))
tree.plot_tree(dtree, feature_names=df.columns[:-2], filled=True)
plt.show()


#decision tree for classification problem

data = pd.read_csv("binary_logistic_regression_data_simulated_for_ML.csv")
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
print(y)
dtree = DecisionTreeRegressor()
plt.figure(figsize=(20,10))
dtree.fit(x,y)
tree.plot_tree(dtree,feature_names=data.columns[:-1],filled=True)
# plt.show()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)
scaled = StandardScaler()
train_scale  = scaled.fit_transform(x_train)
test = scaled.fit_transform(x_test)
print(train_scale)
classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)
# y_value = classifier.fit(x_test,y_test)
# print(y_value)
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
# plt.figure(figsize=(20,10))
# tree.plot_tree(classifier)
# plt.show()


