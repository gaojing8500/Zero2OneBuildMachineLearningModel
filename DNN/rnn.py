class RNNBase(object):
    def __init__(self,description = None):
        self.description = description

    def __str__(self):
        return self.description


class LSTM(RNNBase):
    def __init__(self):
        self._description = "LSTM algorithm"
        RNNBase.__init__(self,self._description)


class GRU(RNNBase):
    def __init__(self):
        self._description = "gru algorithm"
        RNNBase.__init__(self,self._description)

