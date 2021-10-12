'''
Author: your name
Date: 2021-06-18 16:34:54
LastEditTime: 2021-07-13 17:59:07
LastEditors: Please set LastEditors
Description: In User Settings Edit https://github.com/datawhalechina/dive-into-cv-pytorch/blob/master/code/chapter06_transformer
FilePath: /Zero2OneBuildMachineLearningModel/main.py
'''
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable

from BuildDataSet.word2vec_dataset_utils import  DataPreProcess
from DNN.activation import Sigmoid, Tanh, Relu
from DNN.rnn import LSTM, GRU
from DNN.train_transfomer import first_demo, NoamOpt, LabelSmoothing, greedy_decode_demo
from DNN.transformer import subsequent_mask, make_model, PositionalEncoding, MultiHeadedAttentionTest
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

def test_transforme_torch():
    print("\n-----------------------")
    print("test subsequect_mask")
    temp_mask = subsequent_mask(4)
    print(temp_mask)

    print("\n-----------------------")
    print("test build model")
    tmp_model = make_model(10, 10,2)
    print(tmp_model)

def test_train_transformes():
    first_demo()


def test_transfomer_optimizer():
    opts = [NoamOpt(512, 1, 4000, None),
            NoamOpt(512, 1, 8000, None),
            NoamOpt(256, 1, 4000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.show()


def test_label_smoothing():
    # Example of label smoothing.
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    v = crit(Variable(predict.log()),
             Variable(torch.LongTensor([2, 1, 0])))

    # Show the target distributions expected by the system.
    plt.imshow(crit.true_dist)
    plt.show()


def loss(x):
    crit = LabelSmoothing(5, 0, 0.1)
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                     ])
    # print(predict)
    return crit(Variable(predict.log()),
                    Variable(torch.LongTensor([1]))).data[0]


def test_label_smoothing_loss():
    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.show()


def test_position_embedding():
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()

def test_greed_decode():
    greedy_decode_demo()


def  test_transfomer_mutiattention():
    query = torch.rand(64,12,300)
    key = torch.rand(64,10,300)
    value = torch.rand(64,10,300)
    model = MultiHeadedAttentionTest(hidden_dim=300,h_heads=6,dropout=0.1)
    output = model(query,key,value)
    print(output.shape)

def new_decorate(fun):
    def wrapper_fuc():
        print("hello test_decorate")
        fun()
        print("hello test after")
    return wrapper_fuc

@new_decorate
def test_decorate():
    print("hello 装饰器")





if __name__ == "__main__":
    # testID3Tree()
   # test_build_vocab()
   # test_label_smoothing_loss()
   # test_transfomer_mutiattention()
   # test_train_transformes()
    test_decorate()
