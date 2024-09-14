from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
df  = pd.read_csv("binary_logistic_regression_data_simulated_for_ML.csv")
print(df)
x = df.iloc[:,:-1]
print(x)

