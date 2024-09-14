import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix ,accuracy_score,precision_score,f1_score ,recall_score,roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import  make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns


df = pd.read_csv("heart.csv")
# print(df)
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42,shuffle=True)

pipe =  make_pipeline(StandardScaler(),LogisticRegression())

pipe.fit(x_train,y_train)
y_p = pipe.predict(x_test)

cm = confusion_matrix(y_test,y_p)

cm_df = pd.DataFrame(cm)
sns.heatmap(cm_df,annot=True,cmap="BrBG")
plt.xlabel("Actual value")
# plt.ylabel("predicted value")
# plt.show()

# score = accuracy_score(y_test,y_p)
score = accuracy_score(y_test,y_p)
print("Accuracy score is ",score)


threshold_list =[0.2,0.4,0.5,0.6,0.8,1]
for threshold  in  threshold_list :
    y_pred = (pipe.predict_proba(x_test)[:,1] >=threshold).astype('float')
    print("the confusion matrix for ", threshold)
    cm_t = confusion_matrix(y_test,y_pred)
    print(cm_t)
    score = accuracy_score(y_test,y_pred)
    print('the score is ',score ,'for', threshold)
    cm_tdf = pd.DataFrame(cm)
    print(cm_tdf)



