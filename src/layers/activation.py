import numpy as np

class activation:
    def __init__(self):
        self.output=None
    
    def forward(self,x):
        self.output=np.tanh(x)
        return self.output
    
    def backward(self,d_out):
        return d_out*(1-self.output **2)
    