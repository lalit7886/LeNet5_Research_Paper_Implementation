import numpy as np


def batch_(batch_size,dataset,labels,shuffle=True):
    #checked mannualy and correct result
    num_samples=dataset.shape[0]
    if shuffle:
        indices=np.random.permutation(num_samples)
        dataset=dataset[indices]
        labels=labels[indices]
        
    
    for i in range(0,num_samples,batch_size):
        yield dataset[i:i+batch_size,...],labels[i:i+batch_size]
        
class flatten:
    def __init__(self,batched=True):
        self.original_shape=None
        self.batched=batched
    
    def forward(self,x):
        self.original_shape=x.shape
        if self.batched:
            batch_size=x.shape[0]
            return x.reshape(batch_size,-1)
        else:
            return x.reshape(-1)
    def backward(self,x):
        return x.reshape(self.original_shape)
    
def shape_checking(x1,x2):
    assert x1.shape==x2.shape,f"Shape mismatch of {x1.shape}, {x2.shape}"
    
        
    

        
        