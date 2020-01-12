# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 20:46:39 2018

@author: Tigran
"""

import pandas as pd
import numpy as np

# -------------------------------------- import the dataset
housing = pd.read_csv('C:\\Users\\Tigran\\Desktop\\Intro to DS\\supporting\\handson-ml-master\\datasets\\housing\\housing.csv')

# -------------------------------------- get the first inside about the dataset
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()

# --------------------------------------- get more details about the dataset, explore th data

housing["median_income"].hist()

# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)

# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
housing["income_cat"].value_counts()
housing["income_cat"].hist()

# get the correlations of the dataset
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# --------------------------------------- Prepare the data for Machine Learning algorithms
# dealing with missing data
housing.isnull().sum()

# eliminating samples or features with missing values
# rows with missing values can be droped via the dropna method
housing.dropna()

# we can drop columns that have at least one NaN in any row by setting the axis argument to one
housing.dropna(axis = 1)

# the removal of missing data seems to be a convenient approach, it also comes with certain disadvantages
# for example we may end up removing too many samples, which will make a reliable analysis impossible
# Or, if we remove too many feature columns, we will run the risk of losing valuable information that our
# classifier needs to discriminate between classes.

# imputing missing values
# mean imputation
housing_num = housing.drop('ocean_proximity', axis =1)

from sklearn.preprocessing import Imputer
imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imr = imr.fit(housing_num)
imr.statistics_
X = imr.transform(housing_num)
housing_num = pd.DataFrame(X, columns=housing_num.columns,
                          index = list(housing.index.values))

imr.strategy
# other options for the strategy parameter are median and most_frequent


# check if there are missing values of housing_tr
housing_num.isnull().sum()

# performing one-hot encoding on nominal features
housing_cat = housing[['ocean_proximity']]

# understanding the scikit-learn estimator API
# handling categorical data
#from sklearn.preprocessing import LabelEncoder
#ocean_proximity_le = LabelEncoder()
#housing_cat
#housing_cat_encoded = ocean_proximity_le.fit_transform(housing_cat['ocean_proximity'])
#housing_cat_encoded = pd.DataFrame(housing_cat_encoded)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
housing_cat = ohe.fit_transform(housing_cat).toarray()
ohe.categories_

# an even convenient way to create those dummy features via one-hot encodingis to use the get_dummies
# method implemented in pandas
hh = pd.get_dummies(housing_cat[['ocean_proximity']])

# handling the outliers


# --------------------------------------- split the dataset into train and test
X = housing.drop('median_house_value', axis = 1)
y = housing['median_house_value']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(housing['], test_size=0.2, random_state=42)

# --------------------------------------- bringing features onto the same scale



# --------------------------------------- feature engineering, selecting meaningful features



# ----------------------------------------------------------------
# --------------------------------------- Select and train a model 
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# --------------------------------------- Fine-tune your model
housing = train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = train_set["median_house_value"]

housing[housing.isnull().any(axis=1)].head()
housing.isnull().any(axis=1).sum()
housing.isnull(axis = 1).sum()


