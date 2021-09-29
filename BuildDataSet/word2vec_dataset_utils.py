'''
Author: your name
Date: 2021-09-27 19:40:54
LastEditTime: 2021-09-27 19:43:47
LastEditors: your name
Description: In User Settings Edit
FilePath: /Zero2OneBuildMachineLearningModel/BuildDataSet/word2vec_dataset_utils.py
'''
from collections import defaultdict
from typing import List

from torch.utils.data import Dataset

from text2vec.tokenizer import Tokenizer


class DataPreProcess(object):
    def __init__(self, sentences: List[str]):
        self.name = "数据预处理"
        self.tokenize = Tokenizer()
        self.sentences_list = sentences
        self.seg_list = self.tokenize.tokenize(self.sentences_list, "jieba")
        self.vocab_frequency = self.get_frequency_()

    def build_vocab(self):
        vocab_ = defaultdict()
        count = 0
        for seg_sentence in self.seg_list:
            for token in seg_sentence:
                if token in vocab_:
                    continue
                vocab_[token] = count
                count += 1
        return vocab_

    def get_frequency(self,sentences:List[str]):
        seg_list = self.tokenize.tokenize(sentences, "jieba")
        vocab_frequency = defaultdict()
        for seg_sentence in seg_list:
            for token in seg_sentence:
                if token in vocab_frequency:
                    vocab_frequency[token] += 1
                else:
                    vocab_frequency[token] = 1
        return vocab_frequency,seg_list

    def get_frequency_(self):
        vocab_frequency = defaultdict()
        for seg_sentence in self.seg_list:
            for token in seg_sentence:
                if token in vocab_frequency:
                    vocab_frequency[token] += 1
                else:
                    vocab_frequency[token] = 1
        return vocab_frequency



    def build_cbow_dataset(self):
        vocab = {}
        for sentence in self.sentences:
            seg_list = self.tokenize.tokenize(sentence, "jieba")
            for token in seg_list:
                if token in vocab:
                    pass

    def build_skip_gram_dataset(self,sentences: List[str]):
        pass


class Word2VecDataSetUtils(Dataset):
    def __init__(self):
        self.name = ""

    def __str__(self):
        return self.name

    def __getitem__(self):
        pass

    def __len__(self):
        pass
