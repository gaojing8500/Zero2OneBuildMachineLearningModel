'''
Author: your name
Date: 2021-06-18 16:34:54
LastEditTime: 2021-07-13 17:59:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Zero2OneBuildMachineLearningModel/main.py
'''
import numpy as np

from DNN.activation import Sigmoid, Tanh, Relu
from DNN.rnn import LSTM, GRU
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


def testActivateFunctionDNN():
    x = np.arange(-5,5,0.01)
    sigmoid = Sigmoid()
    print(sigmoid)
    sigmoid.show(x)
    tanh = Tanh()
    tanh.show(x)
    print(tanh)
    relu = Relu()
    relu.show(x)
    print(relu)

def testRNN():
    lstm = LSTM()
    print(lstm)
    gru = GRU()
    print(gru)




if __name__ == "__main__":
    # testID3Tree()
   testRNN()
