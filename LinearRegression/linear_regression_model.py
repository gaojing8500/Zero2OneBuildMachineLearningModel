'''
Author: your name
Date: 2021-06-17 20:31:12
LastEditTime: 2021-06-18 19:28:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Zero2OneBuildMachineLearningModel/LinearRegression/linear_regression_model.py
'''
from base_model import BaseModel
import numpy as np
class LinearRegressionModel(BaseModel):
    def __init__(self):
        self.name = "线性回归模型"

    def fit(self,X,y):
        return "fit model"

    def predict(self,X):
        return "预测"

    def decision_function(self,X,y):

        return "决策函数"

    def initialize_w_b(self,dims):
        w = np.zeros((dims,1))
        b = 0
        return w,b


