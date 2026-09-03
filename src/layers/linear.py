import numpy as np


class Linear_layer:
    def __init__(self,in_shape,n_neurons):

        self.weight=np.random.randn(in_shape,n_neurons)
        self.bias=np.zeros((n_neurons,1))
        
    def forward(self,x):
        self.x=x
        return x @ self.weight + self.bias.T
    
    
    def backward(self,d_upstream):
        self.d_wieghts=self.x.T @ d_upstream
        self.b_bias=np.sum(d_upstream,axis=0,keepdims=True).T
        d_input=d_upstream @ self.d_weights.T
        return d_input
    
        
        
        
        
        