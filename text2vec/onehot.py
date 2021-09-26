from collections import defaultdict
from typing import List, Union

from BuildDataSet.word2vec_dataset_utils import DataPreProcess
from text2vec.tokenizer import Tokenizer


class OneHot(object):
    def __init__(self,sentence_list):
        self.name = 'OneHot'
        self.bag_num = 1
        self.data_pre_process = DataPreProcess(sentence_list)
        self.vocab = self.data_pre_process.build_vocab()
        self.vocab_frequency = defaultdict()
        self.tokenize = Tokenizer()

    def __str__(self):
        return self.name


    def transform(self,sentences:List[str]):
        fre_vocab,seg_list = self.data_pre_process.get_frequency(sentences)
        dict_len = len(self.vocab)
        one_hot = [0] * dict_len
        for seg_words in seg_list:
            # if isinstance(seg_words,str):
            #     index = self.vocab[seg_words]
            #     ## 支持词频法和词袋法
            #     if fre_vocab is None:
            #         one_hot[index] = self.bag_num
            #     else:
            #         one_hot[index] = fre_vocab[seg_words]
            if isinstance(seg_words,list):
                for word in seg_words:
                    ##如果在词库中进行one_hot
                    if word in self.vocab:
                        index = self.vocab[word]
                        if fre_vocab is None:
                            one_hot[index] = self.bag_num
                        else:
                            one_hot[index] = fre_vocab[word]
                    else:
                        self.vocab[word] = len(self.vocab)
        return one_hot


