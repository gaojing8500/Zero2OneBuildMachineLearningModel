'''
Author: your name
Date: 2021-08-03 18:02:10
LastEditTime: 2021-08-04 11:49:57
LastEditors: Please set LastEditors
Description: text 基础表示 one-hot、bow、word2vec表示 
FilePath: /Zero2OneBuildMachineLearningModel/word2vec/base.py
'''

from abc import ABC,abstractmethod


class TextVectorBase(ABC):
    """ 文本表示 NLP基础技术之一

    Args:
        ABC ([type]): [description]
    """ 
    @abstractmethod
    def transform(self):
        pass  
    



