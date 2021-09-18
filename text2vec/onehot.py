'''
Author: your name
Date: 2021-08-04 11:47:06
LastEditTime: 2021-08-04 11:47:31
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Zero2OneBuildMachineLearningModel/text2vec/onehot.py
'''
from collections import defaultdict
from typing import List, Union


class OneHot(object):
    def __init__(self,vocab):

        self.name = 'OneHot'
        self.vocab = vocab
        self.bag_num = 1
        self.vocab_frequency = defaultdict()

    def __str__(self):
        return self.name

    def build_vocab(self,texts):
        pass

    def get_frequency(self,sentences:List[str]):
        for token in sentences:
            if token in self.vocab_frequency:
                self.vocab_frequency[token] += 1
            self.vocab_frequency[token] = 1

    def transform(self,words:Union[str,List[str]]):
        dict_len = len(self.vocab)
        one_hot = [0] * dict_len
        if isinstance(words,str):
            index = self.vocab[words]
            one_hot[index] = self.bag_num
        if isinstance(words,list):
            for word in words:
                index = self.vocab[word]
                one_hot[index] = self.bag_num
        return one_hot


