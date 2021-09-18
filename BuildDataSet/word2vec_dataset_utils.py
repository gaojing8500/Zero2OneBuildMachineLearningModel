from collections import defaultdict
from typing import List

from torch.utils.data import Dataset

from text2vec.tokenizer import Tokenizer

tokenize = Tokenizer()




def build_vocab(sentences: List[str]):
    vocab_ = defaultdict()
    count = 0
    for sentence in sentences:
        seg_list = tokenize.tokenize(sentence, "jieba")
        seg_list = seg_list.split(",")
        for token in seg_list:
            if token in vocab_:
                continue
            vocab_[token] = count
            count += 1
    return vocab_

def build_cbow_dataset(sentences: List[str]):
    vocab = {}
    for sentence in sentences:
        seg_list = tokenize.tokenize(sentence, "jieba")
        for token in seg_list:
            if token in vocab:
                pass


def build_skip_gram_dataset(sentences: List[str]):
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
