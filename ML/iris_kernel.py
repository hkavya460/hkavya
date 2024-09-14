from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from  sklearn import svm
import matplotlib.pyplot as  plt
from sklearn import datasets
import warnings
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

df = pd.DataFrame(iris.data)
x = df.iloc[:,0:-2]
y = pd.DataFrame(iris.target)
# print(y)

# fig,ax = plt.subplots(figsize=(3,4))
# scatter =ax.scatter(iris.data[:,0],iris.data[:,1],c=iris.target)
# ax.set(xlabel=iris.feature_names[0],ylabel=iris.feature_names[1])
# _= ax.legend(scatter.legend_elements()[0],iris.target_names,loc="upper right",title= "classes")
# plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,shuffle=True,random_state=42)
# m = svm.SVC(kernel="poly",gamma=20)
# m = LogisticRegression(C=1e5)
m = svm.SVC(kernel="rbf",gamma=0.3,C=0.1)
# m = svm.SVC(kernel="linear",C=8000)
m.fit(x_train,y_train)


fig,ax = plt.subplots(figsize=(4,3))
scatter = ax.scatter(x_train.iloc[:,0],x_train.iloc[:,1],c=y_train,s =50,label=y_train ,edgecolors="k")
ax.set(xlabel=iris.feature_names,ylabel=iris.target_names)
# ax.legend(scatter.legend_elements(),iris.taget_names,loc="upper right",title="classes")
ax = plt.gca()

#
# DecisionBoundaryDisplay.from_estimator(m,x_train,cmap=plt.cm.Paired,ax=ax,response_method="predict",
#                                      plot_method="pcolormesh",
  #                                        shading="auto",
#                                        xlabel="shape lenght",
#                                        ylabel="sepal widht",eps=0.5,)

DecisionBoundaryDisplay.from_estimator(m,x_train,plot_method="contour",colors="k",labels=[-1,0,1],alpha=0.5,linestyle=["__","__"],ax=ax,)
plt.show()
y_pred = m.predict(x_test)
score = accuracy_score(y_test,y_pred)
print(score)
