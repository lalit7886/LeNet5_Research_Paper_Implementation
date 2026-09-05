import numpy as np
from src.utils import one_hot
class cross_entropy:
    def __init__(self):
        self.loss=None
        self.y=None
        self.y_pred=None
        
    def forward(self,y,y_pred):
        self.y=one_hot(y,10)
        self.y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        batch=y.shape[0]
        self.loss= -np.sum(self.y*np.log(self.y_pred))/batch
        return self.loss
    
    def backward(self,d_out):
        batch_size=d_out.shape[0]
        return -(self.y/self.y_pred)/batch_size