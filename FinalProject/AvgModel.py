import os
import json
import argparse

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

#for timestamp column
from datetime import date

# matplotlib and seaborn for plotting
# %matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

# hide warnings
import warnings
def ignore_warn(*args, **kwargs): pass
warnings.warn = ignore_warn

# machine learining
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

# Statistics
from scipy import stats
from scipy.stats import norm, skew #for some statistics

# Mathematics
from math import log

# Pandas options
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns',500 )
pd.set_option('display.max_rows',100 )
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
# pd.reset_option("display.max_rows")

# Save and Load Machine Learning Models
import pickle

from StatisticData import  descriptive_statistics

########################################################
########################################################
# some auxiliary functions
def detect_outliers(df,n,features):
        """
        Takes a dataframe df of features and returns a list of the indices
        corresponding to the observations containing more than n outliers according
        to the Tukey method.
        """
        outlier_indices = []
        # iterate over features(columns)
        for col in features:
            # 1st quartile (25%)
            Q1 = np.percentile(df[col], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(df[col],75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = 1.5 * IQR

            # Determine a list of indices of outliers for feature col
            outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

            # append the found outlier indices for col to the list of outlier indices 
            outlier_indices.extend(outlier_list_col)

        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)        
        multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
        return multiple_outliers
    
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
def plot_feature_importances(df):
        """
        Plot importances returned by a model. This can work with any measure of
        feature importance provided that higher importance is better. 

        Args:
            df (dataframe): feature importances. Must have the features in a column
            called `features` and the importances in a column called `importance

        Returns:
            shows a plot of the 15 most importance features

            df (dataframe): feature importances sorted by importance (highest to lowest) 
            with a column for normalized importance
            """

        # Sort features according to importance
        df = df.sort_values('importance', ascending = False).reset_index()

        # Normalize the feature importances to add up to one
        df['importance_normalized'] = df['importance'] / df['importance'].sum()

        # Make a horizontal bar chart of feature importances
        plt.figure(figsize = (10, 6))
        ax = plt.subplot()

        # Need to reverse the index to plot most important on top
        ax.barh(list(reversed(list(df.index[:15]))), 
                df['importance_normalized'].head(15), 
                align = 'center', edgecolor = 'k')

        # Set the yticks and labels
        ax.set_yticks(list(reversed(list(df.index[:15]))))
        ax.set_yticklabels(df['feature'].head(15))

        # Plot labeling
        plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
        plt.show()
        return df

########################################################
########################################################
def load_data(csv_name, load_from_disk = False):
    #if load_from_disk:
    #    return pd.read_pickle('C:\Users\User\Kaggle\Final_Project\' + csv_name + '.pkl')
    
    data = pd.read_csv('C:/Users/User/Kaggle/Final_Project/' + csv_name + '.csv')
    #data.to_pickle(csv_name + ".pkl")
    return data
########################################################
########################################################
def data_preprocessing(data, is_train_dataset = False):
#     print('---------------------------------------')
#     print('Starting preprocessing')
#     print('---------------------------------------')
#     print('checking the volume of missing values')
#     print('---------------------------------------')
#     missing_values_table(data)
#     print('---------------------------------------')
#     data.drop("Id", axis = 1, inplace = True)
    if is_train_dataset:
#         print('Delete outliers for training dataset')
#         print('---------------------------------------')
        ddxk = descriptive_statistics(df=data,label='SalePrice' )
        Outliers_to_drop = detect_outliers(data,2, ddxk.most_correlated_features(print_heatmap = False))
        data = data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
    
    data["PoolQC"] = data["PoolQC"].fillna("None")
    data["MiscFeature"] = data["MiscFeature"].fillna("None")
    data["Alley"] = data["Alley"].fillna("None")
    data["Fence"] = data["Fence"].fillna("None")
    data["FireplaceQu"] = data["FireplaceQu"].fillna("None")
    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        data[col] = data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        data[col] = data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        data[col] = data[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        data[col] = data[col].fillna('None')
    data["MasVnrType"] = data["MasVnrType"].fillna("None")
    data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
    data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
    data = data.drop(['Utilities'], axis=1)
    data["Functional"] = data["Functional"].fillna("Typ")
    data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
    data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
    data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
    data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
    data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
    data['MSSubClass'] = data['MSSubClass'].fillna("None")
    
    #MSSubClass=The building class
    data['MSSubClass'] = data['MSSubClass'].apply(str)

    #Changing OverallCond into a categorical variable
    data['OverallCond'] = data['OverallCond'].astype(str)


    #Year and month sold are transformed into categorical features.
    data['YrSold'] = data['YrSold'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)

    from sklearn.preprocessing import LabelEncoder
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(data[c].values)) 
        data[c] = lbl.transform(list(data[c].values))
    
    # Adding total sqfootage feature 
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    
    data = pd.get_dummies(data)

    # shape        
#     print('Shape data: {}'.format(data.shape))
#     print('---------------------------------------')
#     print('The preprocessing stage is already done')
#     print('---------------------------------------')
#     print('checking the volume of missing values again')
#     print('---------------------------------------')
#     missing_values_table(data)    
#     print('---------------------------------------')
    return data
########################################################
########################################################
# for training data
# for training data
def label_distribution(data, label):
    sns.distplot(data[label] , fit=norm);

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data[label])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title(label + ' distribution')

    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(data[label], plot=plt)
    plt.show()
########################################################
########################################################
def log_transformation(data, label, show_graph = False):
    #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    data[label] = np.log1p(data[label])
    if show_graph:
        #Check the new distribution 
        sns.distplot(data[label] , fit=norm);

        # Get the fitted parameters used by the function
        (mu, sigma) = norm.fit(data[label])
        print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

        #Now plot the distribution
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                    loc='best')
        plt.ylabel('Frequency')
        plt.title(label + ' distribution')

        #Get also the QQ-plot
        fig = plt.figure()
        res = stats.probplot(data[label], plot=plt)
        plt.show()
########################################################
########################################################
#Validation function
n_folds = 5
def rmsle_cv(model, X, Y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)
    rmse= np.sqrt(-cross_val_score(model, X.values, Y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
########################################################
########################################################
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1) 
########################################################
########################################################
def init():
    lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor()
    model_xgb = xgb.XGBRegressor()
    model_lgb = lgb.LGBMRegressor(objective='regression')
    averaged_models = AveragingModels(models = (lasso , ENet , GBoost, model_xgb, model_lgb))
    train_data = load_data(csv_name='train', load_from_disk=False)
    train_data = data_preprocessing(data=train_data,is_train_dataset=True)
#     label_distribution(train_data, label='SalePrice')
    log_transformation(data=train_data, label='SalePrice', show_graph=False)
    X = train_data.drop(columns=["SalePrice"])
    Y = train_data["SalePrice"]
    averaged_models.fit(X, Y)
    np.save('col.npy', X.columns)
    print("training has been completed succesfully !!!!")
    print("--------------------------------------------")
    filename = 'finalized_model.sav'
    pickle.dump(averaged_models, open(filename, 'wb'))
    return averaged_models
########################################################
########################################################
# Here supposed that test_X is a dataframe with 1 row, for which we must make a prediction
def prediction(model, test_X):
    train_col = np.load('col.npy')
#     test_X = data_preprocessing(data=test_X, is_train_dataset=False)
    missing_cols = set(train_col) - set(test_X.columns )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        test_X[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    test_X = test_X[train_col]
    y_pred = model.predict(test_X)[0]    
    return np.expm1(y_pred)
########################################################
########################################################    
