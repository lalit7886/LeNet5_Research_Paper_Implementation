import numpy as np


class Linear_layer:
    def __init__(self,in_shape,n_neurons):

        self.weight=np.random.randn(in_shape,n_neurons)
        self.bias=np.zeros((n_neurons,1))
        
    def forward(self,x):
        return x @ self.weight + self.bias.T
        
        