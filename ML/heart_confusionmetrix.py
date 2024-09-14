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
#####################################################################
#For the heart.csv dataset, build a logistic regression classifier to predict the risk of heart disease.
df = pd.read_csv("heart.csv")
print(df)
# print(df.describe())
corr_matrix  =[]
corr_matrix = df.corr()
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]
print(x.shape)
print(y.shape)
xc = x.drop(['ca','oldpeak'],axis=1)
# plt.scatter(x ,y)
fig = plt.subplots(figsize=(7,3))
sns.boxplot(df)
plt.show()
corr_matrix = df.corr()
sns.heatmap(corr_matrix,annot=True,cmap="BrBG",center=0,square=True)
plt.show()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42,shuffle=True)

#makepipeline  - allows you to sequentially apply a list of transformers to preprocess the data and, if desired, conclude the sequence with a final predictor for predictive modeling.
pipe =  make_pipeline(StandardScaler(),LogisticRegression())
# Pipeline(steps=[('standard scalar',StandardScaler()),
#         ('logistic regression',LogisticRegression())])
pipe.fit(x_train,y_train)
y_p = pipe.predict(x_test)

score = accuracy_score(y_test,y_p)
print("Accuracy score is  using sklearn is ",score)
def calculate_merices(y_test,y_p,threshold):
    y_var = [1 if y_p > threshold  else 0]


cm_t = []
threshold_list =[0.2,0.4,0.5,0.6,0.8,1]

for threshold in threshold_list :
    y_pred = (pipe.predict_proba(x_test)[:,1]  >= threshold).astype('float')
    cm = confusion_matrix(y_test, y_pred)
    cm_t.append(cm)
print(cm_t)



##Vary the threshold to generate multiple confusion matrices.


#threshold value

accuracy score = Tp +TN / TP +TN+FP+FN
    accuracy = (df_t.iloc [0,0] +df_t.iloc [1,0])  /  df_t.sum().sum()
    print("accuracy score for", threshold ,'is', accuracy)
    sns.heatmap(cm, annot=True, cmap="BrBG")
    plt.xlabel("Actual value")
    plt.ylabel("predicted value")
    plt.show()
Recall score =  True Positives / (False Negatives + True Positives)
    recall = (df_t.iloc[0,0])  / (df_t.iloc[0,0] + df_t.iloc[1,0])
    print("recall score is ", recall)
    # Precision Score = True Positives/ (False Positives + True Positives)
    precision = (df_t.iloc[0,0])  / (df.iloc[0,0] + df_t.iloc[0,1])
    print("precision score is ", precision)
    #F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score)
    f1score = 2  * precision * recall
    print("F1 score is ",f1score )






    # print(df_t)
    # df_t.columns = ['True positive','False positive']
    # df_t.index = ['']
    # fpr, tpr, threshold = roc_curve(y_test, y_p)
    #
    # cm_df = pd.DataFrame({'FPR':fpr,'TPR':tpr,'Thresholds':threshold})
    #
    #
    # print(cm_df)
    #


    # fpr ,tpr,threshold= roc_curve(y_test,y_p)
    # res = pd.DataFrame({'FPR':fpr,'TPR':tpr,'Thresholds':threshold})
    # print(res)

# score = accuracy_score(y_test,y_p)
# print("Accuracy score is ",score)
#
# cm = confusion_matrix(y_test,y_p)
# cm_df = pd.DataFrame(cm)
# sns.heatmap(cm_df,annot=True,cmap="BrBG")
# plt.xlabel("Actual value")
# plt.ylabel("predicted value")

# fig ,ax= plt.subplots(figsize=(4,3))
# scatter= ax.scatter(x_train.iloc[:,0],x_train.iloc[:,1],c=y_train,label= y_train,edgecolors="k")
# ax =plt.gca
# DecisionBoundaryDisplay.from_estimator(m,x_train,plot_method="contour",colors="k",labels=[-1,0,1],alpha=0.1,linestyle=["__","__"],ax=ax,)
# plt.show()


#Accuracy : True positive + False positive / Total
# Precision Score = True Positives/ (False Positives + True Positives)
precisionscore = precision_score(y_test,y_p)
print("precision score  using sklearn is",precisionscore)

#recall score
# Recall Score = True Positives / (False Negatives + True Positives)
recall_score = recall_score (y_test,y_p)
print("recall score  using sklearn is ",recall_score)

# f1 score
#F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score)
f1score = f1_score(y_test,y_p)
print("f1 score using sklearn  is ",f1score)

#ROC curve
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.30,random_state=1,shuffle =True)
pipe = make_pipeline(StandardScaler(),LogisticRegression())
pipe.fit(x_train.iloc[:,[2,12]],y_train)
y_pr = pipe.predict(x_test.iloc[:,[2,12]])
fpr1,tpr1,threshold =roc_curve(y_test,y_pr,pos_label=1)
roc_auc1 = auc(fpr1,tpr1)
pipe.fit(x_train.iloc[:,[1,-1]],y_train)
y_pre = pipe.predict(x_test.iloc[:,[1,-1]])
fpr2,tpr2,thresholds = roc_curve(y_test,y_pre,pos_label=1)
roc_auc2 = auc(fpr2,tpr2)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
plt.plot(fpr1, tpr1, label='ROC Curve 1 (AUC = %0.2f)' % (roc_auc1))
plt.plot(fpr2, tpr2, label='ROC Curve 2 (AUC = %0.2f)' % (roc_auc2))
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier')
plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='green', label='Perfect Classifier')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")
plt.show()
#from scratch


