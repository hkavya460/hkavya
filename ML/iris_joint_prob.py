from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Develop prediction model for Iris.csv using joint probability distribution approach:
#   (a) Use only first two features, SepalLengthCm, SepalWidthCm
#   (b) Add random noise to the features
#   (c) Discretize the feature values
#   (d) Build a decision tree model with max_depth = 2, then, compare the accuracy of this model with the joint probability distribution method

def get_iris():
    df = pd.read_csv(r"/home/ibab/datasets/Iris.csv")
    x = df.iloc[:,1:3]
    y = df.iloc[:, -1]
    return df, x, y

def add_random_noise(x):
    noise = np.random.normal(0, 0.1, x.shape)
    x = x + noise
    return x

def make_features_discrete(x):
    for col in x.columns:
        x[col] = x[col].astype(int)
    return x

def fit_model(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def frequency_table(df):
    table = df.groupby([df.columns[0], df.columns[1], df.columns[2]]).size().reset_index(name='Probability')
    table["Probability"] = table["Probability"] / len(df.index)
    idx = table.groupby(['SepalLengthCm', 'SepalWidthCm'])['Probability'].idxmax()
    filtered_table = table.loc[idx]
    return filtered_table

def predict_class(df_row_values, table):
    x_predicted = table.loc[ (table[table.columns[0]] == df_row_values[0]) & (table[table.columns[1]] == df_row_values[1]), 'Species']
    if len(x_predicted) == 0:
        return 0
    return x_predicted.iloc[0]

def joint_probability_distribution(df):
    train_size = int(0.7 * len(df))     # Divide data into train and test
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    table = frequency_table(df_train)       # Frequency table
    count = 0
    correct = 0
    for i in range(len(df_test.index)):
        predicted_class = predict_class(df_test.iloc[i], table)
        if predicted_class == df_test.iloc[i, 2]:
            correct = correct + 1
        count = count + 1
    return correct/count        # Percentage of correct predictions

def main():
    df, x, y = get_iris()       # Load data
    x = add_random_noise(x)     # Add random noise
    x = make_features_discrete(x)       # Make feature values int (change from continuous to discrete)
    df = pd.concat([x, y], axis=1)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Joint probability distribution method
    accuracy_jpd = joint_probability_distribution(df)
    print("Accuracy using Joint Probability Distribution", accuracy_jpd)

    # Decision Tree
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
    model = DecisionTreeClassifier(max_depth=2)
    accuracy_decision_tree = fit_model(x_train, y_train, x_test, y_test, model)
    print("Decision Tree accuracy: ", accuracy_decision_tree)

    # Gaussian Naive Bayes  # not required
    # model = GaussianNB()
    # accuracy_naive_gaussian = fit_model(x_train, y_train, x_test, y_test, model)
    # print("Naive Gaussian Accuracy: ", accuracy_naive_gaussian)

if __name__ == "__main__":
    main()