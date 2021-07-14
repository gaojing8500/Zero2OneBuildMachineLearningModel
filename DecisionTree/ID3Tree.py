'''
Author: your name 信息增益进行划分
Date: 2021-07-05 19:41:19
LastEditTime: 2021-07-14 21:15:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Zero2OneBuildMachineLearningModel/DecisionTree/D3Tree.py
'''
from numpy.lib.arraysetops import unique
from numpy.lib.shape_base import split
from numpy.lib.utils import info
from sklearn import tree
from base_model import BaseModel

from math import log
import pandas as pd

class ID3Tree(BaseModel):
    def __init__(self):
       self.name = "D3 tree "
    def fit(self, X, y):
        return "shsh"
    def predict(self, X):
        return "shshshsh"
    ##求解信息熵
    def entropy(self,ele):
        probs = [ele.count(i)/len(ele) for i in set(ele)]
        entropy = -sum([prob*log(prob,2) for prob in probs])
        return entropy;
    ##加入某一个特征后的条件熵
    def conditionEntropy(self,ele,conditionEle):
        probs = [conditionEle.count(i)/len(conditionEle) for i in set(conditionEle)]
        conditionEntropy = sum([prob * self.entropy(ele) for prob in probs])
        return conditionEntropy

    def preprocessData(self,data,col):
        unique_valuse = data[col].unique()
        result_dict = {elem:pd.DataFrame for elem in unique_valuse}
        for key in result_dict.keys():
            result_dict[key] = data[:][data[col] == key]
        return result_dict

    def choose_best_col(self,df,label):
        info_gain_list = []
        entropy_D = self.entropy(df[label].tolist())
        cols = [col for col in df.columns if col not in [label]]
        max_value,best_col = -999,None
        max_splited = None
        for col in cols:
            split_set = self.preprocessData(df,col)
            entropy_DA = 0
            for subset_col, subset in split_set.items():
                entropy_Di = self.entropy(subset[label].tolist())
                entropy_DA += len(subset)/len(df) * entropy_Di
            ##计算信息增益
            info_gain = entropy_D - entropy_DA
            info_gain_list.append(info_gain)
        info_gain_list.sort()
        return info_gain_list




        


# clf = tree.DecisionTreeClassifier(criterion = 'entropy',splitter='best')


