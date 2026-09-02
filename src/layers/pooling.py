import numpy as np

class average_pooling:
    def __init__(self,pool_size=2,stride=2):
        self.stride=stride
        self.pool=pool_size
    
    def average_pooing(self,x:np.array):
        batch,n_fiters,height,width=x.shape
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