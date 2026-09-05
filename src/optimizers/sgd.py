import numpy as np

class SGD:
    def __init__(self,learning_rate=0.0005):
        self.learning_rate=learning_rate
        
    def update(self,w,dw,b,db):
        w-=self.learning_rate*dw
        b-=self.learning_rate*db
        return w,b
    