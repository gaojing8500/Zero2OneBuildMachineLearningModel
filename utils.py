'''
Author: your name
Date: 2021-06-18 15:34:12
LastEditTime: 2021-06-18 17:32:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Zero2OneBuildMachineLearningModel/utils.py
'''
from abc import ABCMeta,abstractmethod

class ActivateFuctionBase(metaclass=ABCMeta):
    def __str__(self):
        return "激活函数抽象类"
    def __call__(self,x):
        return "对应子类实现"

class Softmax(ActivateFuctionBase):
    def __init__(self):
        self.name = "softmax function"