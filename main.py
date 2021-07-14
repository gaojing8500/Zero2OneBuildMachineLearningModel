'''
Author: your name
Date: 2021-06-18 16:34:54
LastEditTime: 2021-07-13 17:59:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Zero2OneBuildMachineLearningModel/main.py
'''
from LinearRegression.linear_regression_model import LinearRegressionModel
from DecisionTree.ID3Tree import ID3Tree

import pandas as pd



def testLinearRegression():
    model = LinearRegressionModel()


def testID3Tree():
    model = ID3Tree()
    data = pd.read_csv("example_data.csv")
    print(data["play"].to_list())
    print(model.entropy(data["play"].to_list()))
    print(data["outlook"].to_list())
    print(model.conditionEntropy(data["play"].to_list(),data["outlook"].to_list()))
    model.choose_best_col(data,"play")



if __name__ == "__main__":
    testID3Tree()

