import numpy as np

class tanh:
    def __init__(self):
        self.output=None
    
    def forward(self,x):
        self.output=np.tanh(x)
        return self.output
    
    def backward(self,d_out):
        return d_out*(1-self.output **2)
    
    
class softmax:
    def __init__(self):
        self.output=None
        
    def forward(self,x):
        shifted_x=x-np.max(x,axis=1,keepdims=True)
        exp_x=np.exp(shifted_x)
        self.output=exp_x/np.sum(exp_x,axis=1,keepdims=True)
        return self.output
    
    def backward(self,y_actual):
        batch_size=y_actual.shape[0]
        return (self.output-y_actual)/batch_size
        