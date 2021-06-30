'''
Author: your name
Date: 2021-06-17 20:31:12
LastEditTime: 2021-06-30 14:26:55
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

    def decision_function(self,X,y,w,b):
        num_train = X.shape[0]
        num_feature = X.shape[1]
        ##模型公式
        y_hat = np.dot(X,w) + b
        ##损失函数
        loss = np.sum((y_hat-y)**2)/num_train
        ## 参数的偏导
        dw = np.dot(X.T,(y_hat-y))/num_train
        return "决策函数"

    def initialize_w_b(self,dims):
        w = np.zeros((dims,1))
        b = 0
        return w,b 
    
        
        
        


