import numpy as np

class cross_entropy:
    def __init__(self):
        self.loss=None
        
    def forward(self,y,y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        batch=y.shape[0]
        self.loss= -np.sum(y*np.log(y_pred))/batch
        return self.loss