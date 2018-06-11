
# coding: utf-8

# In[2]:


import numpy as np
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


# In[3]:


class SoftmaxWithLoss:
    #インスタンス変数に取っておく
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        # forwardの式
        # -sum ( t * log (y))
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout=1):
        # backwardの式
        # yi - ti (iはIndex)
        batch_size = self.t.shape[0]
        # Backwardを実装して、微分値をdxに代入してください
        dx = (self.y - self.t) / batch_size
        return dx

