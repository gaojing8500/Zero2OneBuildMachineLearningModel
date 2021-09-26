'''
Author: your name
Date: 2021-06-18 16:34:54
LastEditTime: 2021-07-13 17:59:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Zero2OneBuildMachineLearningModel/main.py
'''
import numpy as np

from BuildDataSet.word2vec_dataset_utils import  DataPreProcess
from DNN.activation import Sigmoid, Tanh, Relu
from DNN.rnn import LSTM, GRU
from EncoderDecoder.Seq2Seq import Encoder, Decoder, Seq2Seq
from LinearRegression.linear_regression_model import LinearRegressionModel
from DecisionTree.ID3Tree import ID3Tree

import pandas as pd

from service.service_tokenizer import TokenizerSerive, tokenizer_service
from text2vec.onehot import OneHot


def testLinearRegression():
    model = LinearRegressionModel()


def testID3Tree():
    model = ID3Tree()
    data = pd.read_csv("example_data.csv")
    print(data["play"].to_list())
    print(model.entropy(data["play"].to_list()))
    print(data["outlook"].to_list())
    print(model.conditionEntropy(data["play"].to_list(),data["outlook"].to_list()))
    model.choose_best_col(data,"play")


def testActivateFunctionDNN():
    x = np.arange(-5,5,0.01)
    sigmoid = Sigmoid()
    print(sigmoid)
    sigmoid.show(x)
    tanh = Tanh()
    tanh.show(x)
    print(tanh)
    relu = Relu()
    relu.show(x)
    print(relu)

def testRNN():
    lstm = LSTM()
    print(lstm)
    gru = GRU()
    print(gru)
import torch

def testSeq2Seq():
    INPUT_DIM = 7853
    OUTPUT_DIM = 5893
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    BATCH_SIZE = 128

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec)

    # def init_weights(m):
    #     for name, param in m.named_parameters():
    #         torch.nn.init.uniform_(param.data, -0.08, 0.08)
    #
    # model.apply(init_weights)
    # print(model)
    src = torch.tensor([[1, 2, 3], [6, 0, 0], [4, 5, 0], [3, 7, 1]])
    tag = torch.tensor([[1, 2, 3], [6, 0, 0], [4, 5, 0], [3, 7, 1]])

    output = model(src,tag)


def tokenizer_serive():
    tokenizer_service.run(host="10.5.69.38",port=8989)





def test_build_vocab():
    sentence = ["去重，字面意思就是去掉重复的东西。在我们写论文时，如果重复率过高，我们的论文是不能通过的。",
                "因此需要通过查重修改，将重复的重复的内容删除更改，这就是查重。在python中，如果一组内容有重复的内容，",
                "他们发挥的作用也是一样的，这时我们就需要去重。你知道在python中如何去重吗？本文小编就告诉你几种去重的方法",
                "武汉联影是一家牛逼的公司"]

    test_list = ["我们是联影的好员工，联影是我们的家。"]
    # data_pre_process = DataPreProcess(sentence)
    # vocab = data_pre_process.build_vocab()
    # fre_vocab = data_pre_process.get_frequency(test_list)
    one_hot = OneHot(sentence)
    # one_hot = one_hot.transform("武汉")
    # print(one_hot)
    # fre = one_hot.get_frequency(sentence)
    one_hot_ = one_hot.transform(["武汉联影是一家牛逼的公司，我们是武汉联影的一员。"])
    print(one_hot_)



if __name__ == "__main__":
    # testID3Tree()
   test_build_vocab()
