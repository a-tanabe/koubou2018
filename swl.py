
# coding: utf-8

# In[ ]:


from common.functions import *
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

