from flask import Flask, abort, jsonify, request, render_template
from urllib import parse
from sklearn.externals import joblib
# import pickle
import AvgModel
import StatisticData
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import os
import json
# import argparse
import numpy as np
import pandas as pd 
###
###
dict = {
    'OverallQual' : 5 ,
    'YearBuilt' : 1961,
    'YearRemodAdd' : 1961,
    'TotalBsmtSF' : 882.000 ,
    '1stFlrSF' : 896,
    'GrLivArea' : 896,
    'FullBath' : 1,
    'TotRmsAbvGrd' : 5,
    'GarageCars' : 1.000,
    'GarageArea' : 730.000
}
df = pd.DataFrame(dict, index=[0])
my_model = AvgModel.init()
app = Flask(__name__)

@app.route("/")
def index():
    return "Hello"

# http://localhost:5000/prediction
@app.route("/prediction")
def predictor1():
	return jsonify(AvgModel.prediction(my_model,df))

# http://localhost:5000/AllParameters
@app.route("/AllParameters")
def getAllParams():
    return pd.DataFrame(StatisticData.df.columns).to_html()

# http://localhost:5000/predictionWithParams?OverallQual=5&YearBuilt=1961&YearRemodAdd=1961&TotalBsmtSF=882.000&1stFlrSF=896&GrLivArea=896&FullBath=1&TotRmsAbvGrd=5&GarageCars=1.000&GarageArea=730.000
@app.route('/predictionWithParams')
def predictionWithParams():
   newDict = {} # an empty dictionary
   for key, value in request.args.to_dict().items():
       newDict[key] = float(value)
   df = pd.DataFrame(newDict, index=[0])
   # df = pd.DataFrame(request.args.to_dict(), index=[0])
   # return df.to_html()
   return jsonify(AvgModel.prediction(my_model, df))

# http://localhost:5000/stat/mean/OverallQual
@app.route("/stat/<string:mean>/<string:column>")
def statMean(mean, column):
    return jsonify(StatisticData.Stat.mean(feature=column))

# http://localhost:5000/stat/median/SalePrice
@app.route("/stat/<string:median>/<string:column>")
def statMedian(median, column):
    return jsonify(StatisticData.Stat.median(feature=column))

# http://localhost:5000/stat/mostCorrelatedFeatures
@app.route("/stat/mostCorrelatedFeatures")
def statMostCorrFeat():
    df = pd.DataFrame(np.delete(StatisticData.Stat.most_correlated_features(print_heatmap=False), 10))
    return df.to_html()

# http://localhost:5000/HouseByStreet/Grvl
@app.route("/HouseByStreet/<string:street>")
def HouseByStreet(street):
    return StatisticData.getHouseByStreet(street).to_html()

# http://localhost:5000/HouseByPrice/100000/110000
@app.route("/HouseByPrice/<int:startPrice>/<int:endPrice>")
def HouseByPrice(startPrice, endPrice):
    return StatisticData.getHouseByPrice(startPrice, endPrice).to_html()

if __name__ == "__main__":
    app.run(debug=True)
