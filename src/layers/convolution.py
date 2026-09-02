import numpy as np

class Conv2D:
    def __init__(self,out_channels=6,in_channels=3,kernel_size=3):
            self.weights=np.random.randn(out_channels,in_channels,kernel_size,kernel_size)
            self.bias=np.zeros((out_channels,1))
            self.out_channels=out_channels
            self.kernel_size=kernel_size

    def forward(self,x):
        batch_size,channel,height,width=x.shape
        output=np.zeros((batch_size,self.out_channels,height-self.kernel_size+1,width-self.kernel_size+1))
        for id,batch in enumerate(x):
            for out_,filter in enumerate(self.weights):
                for height_ in range(height-self.kernel_size+1):
                    for width_ in range(width-self.kernel_size+1):
                        output[id][out_][height_][width_]=np.sum(filter*batch[:,height_:height_+self.kernel_size,width_:width+self.kernel_size
                                                                        ])+self.bias[out_][0]
        return output
    
    
                
            
        
        
        