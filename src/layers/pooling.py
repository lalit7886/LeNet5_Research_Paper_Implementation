import numpy as np

class average_pooling:
    def __init__(self,pool_size=2,stride=2):
        self.stride=stride
        self.pool=pool_size
        self.input_shape=None
        
    
    def forward(self,x:np.array):
        self.input_shape=x.shape
        batch,n_fiters,height,width=self.input_shape
        output_height=((height-self.pool)//self.stride) + 1
        output_width=((width-self.pool) // self.stride) + 1
        output=np.zeros((batch,n_fiters,output_height,output_width))
        for batch_id,batch in enumerate(x):
            for fiter_id, filter in enumerate(batch):
                for height in range(output_height):
                    for width in range(output_width):
                        h_start=height*self.stride
                        w_start=width*self.stride
                        patch=filter[
                            h_start:h_start+self.pool,
                            w_start:w_start+self.pool
                        ]
                        
                        output[batch_id][fiter_id][height][width]=np.mean(patch)
                        
        
        return output
    
    def backward(self,d_out):
        batch,n_kernels,img_height,img_width=d_out.shape
        output_=np.zeros((self.input_shape))
        for batch_id, batch in enumerate(d_out):
            for filter_id, filter in enumerate(batch):
                for height in range(filter.shape[0]):
                    for width in range(filter.shape[1]):
                        start_h=height*self.stride
                        start_w=width*self.stride
                        output_[batch_id,filter_id,start_h:start_h+self.pool,start_w:start_w+self.pool]=0.25*d_out[batch_id][filter_id][height][width]
        
        return output_
    
if __name__ == "__main__":
    x = np.array([[
        [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9,10,11,12],
        [13,14,15,16]]
    ]])

    pool_size=2
    stride=2
    print(f"Result before forward pooling \n {x}")
    pool=average_pooling()
    x=pool.forward(x)
    print(f"Result after forward pooling \n {x}")
    x=pool.backward(x)
    print(f"Result after backward pooling \n {x}")

