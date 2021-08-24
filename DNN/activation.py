import numpy as np
from matplotlib import pyplot as pl


class ActivateFuctionBase(object):
    def __init__(self,description = None):
        self.description = description

    def __str__(self):
        return self.description

    def __call__(self, x):
        pass

    @classmethod
    def show(cls, x):
        result = cls()
        pl.plot(result(x))
        pl.show()


class Sigmoid(ActivateFuctionBase):
    def __init__(self):
        self._description = "sigmoid functions"
        super().__init__(self._description)

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))


class Softmax(ActivateFuctionBase):
    def __init__(self):
        self._description = "sigmoid functions"
        super().__init__(self._description)

    def __call__(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


class Tanh(ActivateFuctionBase):
    def __init__(self):
        self._description = "Tanh functions"
        super().__init__(self._description)

    def __call__(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


class Relu(ActivateFuctionBase):
    def __init__(self):
        self._description = "Relu functions"
        super().__init__(self._description)

    def __call__(self, x):
        return np.where(x < 0, 0, x)
