#! /usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

pwd = os.getcwd()
filepath = os.path.join(pwd, "housing.csv") #cretes a file path to the csv file in the current working directory
# print ("File path: ", filepath)

housing_data = pd.read_csv(filepath)
housing_data
# print(housing_data)
# housing_data.info() #summary of the dataframe

# print(housing_data["ocean_proximity"].value_counts())
# housing_data["ocean_proximity"].value_counts().plot(kind='barh', title='ocean proximity distribution')
# plt.show()


# print(housing_data.describe()) # statistical summary of the dataframe

# housing_data["median_income"].hist()
# plt.title("Median Income Distribution")
# plt.xlabel("Median Income")
# plt.ylabel("Count")
# plt.show()

housing_data["income_cat"] = pd.cut(housing_data["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                                    labels=[1, 2, 3, 4, 5])
# print(housing_data["income_cat"].value_counts())

housing_data["income_cat"].hist()
# plt.show()

y = housing_data["median_house_value"]
X = housing_data.drop("median_house_value", axis=1)
# print(X)

#Split Datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33) #train test split that splits data horizontally into a training set and a testing set
# print("X_train shape: ", X_train.shape)

housing_data["income_cat"].hist
# plt.show()

X_train["income_cat"].hist
# plt.show()

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
    strat_train_set = housing_data.loc[train_index]
    strat_test_set = housing_data.loc[test_index]

print(strat_train_set)