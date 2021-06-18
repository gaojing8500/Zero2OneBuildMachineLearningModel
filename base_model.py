'''
Author: your name
Date: 2021-06-17 20:51:49
LastEditTime: 2021-06-18 16:30:00
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Zero2OneBuildMachineLearningModel/base_model.py
'''

from abc import ABCMeta,abstractmethod

class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def fit(self,X,y):
        pass
    @abstractmethod
    def predict(self,X):
        pass