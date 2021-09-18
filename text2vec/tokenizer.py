from typing import List
import jieba
import json
##hanlp还是比较麻烦的
# from pyhanlp import *

class Tokenizer(object):
    def __init__(self):
        self.name = "Jieba hanlp allennlp corenlp"

    def __str__(self):
        return self.name

    def tokenize(self, sentences, tokenize_label):
        if tokenize_label == "jieba":
            if isinstance(sentences, List):
                pass
            if isinstance(sentences,str):
                seg_list = jieba.cut(sentences,cut_all=False)
                return ",".join(seg_list)
        if tokenize_label == "hanlp":
            if isinstance(sentences, str):
                return None




