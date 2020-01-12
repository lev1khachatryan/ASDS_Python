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

# from  AvgModel import load_data

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
class descriptive_statistics:
    def __init__(self, df, label):
        self.df = df
        self.label = label
        
    def corrmat(self):
        return self.df.corr()
        
    def most_correlated_features(self, print_heatmap = True ,tol=0.5):
        corrmat = self.df.corr()
        top_corr_features = corrmat.index[abs(corrmat[self.label]) > tol]
        if print_heatmap:
            plt.figure(figsize=(10,10))
            sns.heatmap(self.df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
        return top_corr_features.values
    
#     def graph_between_most_correlated_features(self, tol=0.5):
#         sns.set()
#         corrmat = self.df.corr()
#         top_corr_features = corrmat.index[abs(corrmat[self.label]) > tol]
# #         corrmat.index[abs(corrmat[self.label]) > tol]
#         cols = top_corr_features.values
#         sns.pairplot(self[cols], size = 2.5)
#         plt.show();
    
    def mean(self, feature):
        return np.mean(self.df[feature])
    
    def median(self, feature):
        return np.median(self.df[feature])
    
    def standard_deviation(self, feature):
        return np.std(self.df[feature])
########################################################
########################################################
df = load_data(csv_name='train', load_from_disk=False)
Stat = descriptive_statistics(df=df, label='SalePrice')
########################################################
########################################################
def getHouseByPrice(startPrice , endPrice, df = df):
    return df.loc[(df['SalePrice'] >= startPrice) & (df['SalePrice'] <= endPrice)]
########################################################
########################################################
def getHouseByStreet(street, df = df):
    return df.loc[df["Street"] == street]
########################################################
########################################################
# train_col = ['Id' 'MSSubClass' 'LotFrontage' 'LotArea' 'Street' 'Alley' 'LotShape'
#  'LandSlope' 'OverallQual' 'OverallCond' 'YearBuilt' 'YearRemodAdd'
#  'MasVnrArea' 'ExterQual' 'ExterCond' 'BsmtQual' 'BsmtCond' 'BsmtExposure'
#  'BsmtFinType1' 'BsmtFinSF1' 'BsmtFinType2' 'BsmtFinSF2' 'BsmtUnfSF'
#  'TotalBsmtSF' 'HeatingQC' 'CentralAir' '1stFlrSF' '2ndFlrSF'
#  'LowQualFinSF' 'GrLivArea' 'BsmtFullBath' 'BsmtHalfBath' 'FullBath'
#  'HalfBath' 'BedroomAbvGr' 'KitchenAbvGr' 'KitchenQual' 'TotRmsAbvGrd'
#  'Functional' 'Fireplaces' 'FireplaceQu' 'GarageYrBlt' 'GarageFinish'
#  'GarageCars' 'GarageArea' 'GarageQual' 'GarageCond' 'PavedDrive'
#  'WoodDeckSF' 'OpenPorchSF' 'EnclosedPorch' '3SsnPorch' 'ScreenPorch'
#  'PoolArea' 'PoolQC' 'Fence' 'MiscVal' 'MoSold' 'YrSold' 'TotalSF'
#  'MSZoning_C (all)' 'MSZoning_FV' 'MSZoning_RH' 'MSZoning_RL'
#  'MSZoning_RM' 'LandContour_Bnk' 'LandContour_HLS' 'LandContour_Low'
#  'LandContour_Lvl' 'LotConfig_Corner' 'LotConfig_CulDSac' 'LotConfig_FR2'
#  'LotConfig_FR3' 'LotConfig_Inside' 'Neighborhood_Blmngtn'
#  'Neighborhood_Blueste' 'Neighborhood_BrDale' 'Neighborhood_BrkSide'
#  'Neighborhood_ClearCr' 'Neighborhood_CollgCr' 'Neighborhood_Crawfor'
#  'Neighborhood_Edwards' 'Neighborhood_Gilbert' 'Neighborhood_IDOTRR'
#  'Neighborhood_MeadowV' 'Neighborhood_Mitchel' 'Neighborhood_NAmes'
#  'Neighborhood_NPkVill' 'Neighborhood_NWAmes' 'Neighborhood_NoRidge'
#  'Neighborhood_NridgHt' 'Neighborhood_OldTown' 'Neighborhood_SWISU'
#  'Neighborhood_Sawyer' 'Neighborhood_SawyerW' 'Neighborhood_Somerst'
#  'Neighborhood_StoneBr' 'Neighborhood_Timber' 'Neighborhood_Veenker'
#  'Condition1_Artery' 'Condition1_Feedr' 'Condition1_Norm'
#  'Condition1_PosA' 'Condition1_PosN' 'Condition1_RRAe' 'Condition1_RRAn'
#  'Condition1_RRNe' 'Condition1_RRNn' 'Condition2_Artery'
#  'Condition2_Feedr' 'Condition2_Norm' 'Condition2_PosA' 'Condition2_RRAe'
#  'Condition2_RRAn' 'Condition2_RRNn' 'BldgType_1Fam' 'BldgType_2fmCon'
#  'BldgType_Duplex' 'BldgType_Twnhs' 'BldgType_TwnhsE' 'HouseStyle_1.5Fin'
#  'HouseStyle_1.5Unf' 'HouseStyle_1Story' 'HouseStyle_2.5Fin'
#  'HouseStyle_2.5Unf' 'HouseStyle_2Story' 'HouseStyle_SFoyer'
#  'HouseStyle_SLvl' 'RoofStyle_Flat' 'RoofStyle_Gable' 'RoofStyle_Gambrel'
#  'RoofStyle_Hip' 'RoofStyle_Mansard' 'RoofStyle_Shed' 'RoofMatl_CompShg'
#  'RoofMatl_Membran' 'RoofMatl_Metal' 'RoofMatl_Roll' 'RoofMatl_Tar&Grv'
#  'RoofMatl_WdShake' 'RoofMatl_WdShngl' 'Exterior1st_AsbShng'
#  'Exterior1st_AsphShn' 'Exterior1st_BrkComm' 'Exterior1st_BrkFace'
#  'Exterior1st_CBlock' 'Exterior1st_CemntBd' 'Exterior1st_HdBoard'
#  'Exterior1st_ImStucc' 'Exterior1st_MetalSd' 'Exterior1st_Plywood'
#  'Exterior1st_Stone' 'Exterior1st_Stucco' 'Exterior1st_VinylSd'
#  'Exterior1st_Wd Sdng' 'Exterior1st_WdShing' 'Exterior2nd_AsbShng'
#  'Exterior2nd_AsphShn' 'Exterior2nd_Brk Cmn' 'Exterior2nd_BrkFace'
#  'Exterior2nd_CBlock' 'Exterior2nd_CmentBd' 'Exterior2nd_HdBoard'
#  'Exterior2nd_ImStucc' 'Exterior2nd_MetalSd' 'Exterior2nd_Other'
#  'Exterior2nd_Plywood' 'Exterior2nd_Stone' 'Exterior2nd_Stucco'
#  'Exterior2nd_VinylSd' 'Exterior2nd_Wd Sdng' 'Exterior2nd_Wd Shng'
#  'MasVnrType_BrkCmn' 'MasVnrType_BrkFace' 'MasVnrType_None'
#  'MasVnrType_Stone' 'Foundation_BrkTil' 'Foundation_CBlock'
#  'Foundation_PConc' 'Foundation_Slab' 'Foundation_Stone' 'Foundation_Wood'
#  'Heating_Floor' 'Heating_GasA' 'Heating_GasW' 'Heating_Grav'
#  'Heating_OthW' 'Heating_Wall' 'Electrical_FuseA' 'Electrical_FuseF'
#  'Electrical_FuseP' 'Electrical_Mix' 'Electrical_SBrkr'
#  'GarageType_2Types' 'GarageType_Attchd' 'GarageType_Basment'
#  'GarageType_BuiltIn' 'GarageType_CarPort' 'GarageType_Detchd'
#  'GarageType_None' 'MiscFeature_Gar2' 'MiscFeature_None'
#  'MiscFeature_Othr' 'MiscFeature_Shed' 'MiscFeature_TenC' 'SaleType_COD'
#  'SaleType_CWD' 'SaleType_Con' 'SaleType_ConLD' 'SaleType_ConLI'
#  'SaleType_ConLw' 'SaleType_New' 'SaleType_Oth' 'SaleType_WD'
#  'SaleCondition_Abnorml' 'SaleCondition_AdjLand' 'SaleCondition_Alloca'
#  'SaleCondition_Family' 'SaleCondition_Normal' 'SaleCondition_Partial']
########################################################
########################################################
