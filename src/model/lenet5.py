from src.layers.activation import tanh,softmax
from src.layers.convolution import Conv2D
from src.layers.linear import Linear_layer
from src.layers.pooling import average_pooling
from src.losses.cross_entropy import cross_entropy
from src.optimizers.sgd import SGD
from src.utils import flatten

class LeNet5:
    def __init__(self):
        self.Convolution1=Conv2D()
        self.tanh_act1=tanh()
        self.average_pooling1=average_pooling()
        
        self.Convolution2=Conv2D(16,6,5)
        self.tanh_act2=tanh()
        self.average_pooling2=average_pooling()
        
        self.flatten=flatten()
        self.Linear_layer1=Linear_layer(400,84)
        self.tanh_act3=tanh()
        self.Linear_layer2=Linear_layer(84,10)
        self.softmax=softmax()
        self.optimizer=SGD()
        
  
        self.loss_=cross_entropy()
        self.optimizer=SGD()
        
    def forward(self,x):
        x=self.Convolution1.forward(x)
        x=self.tanh_act1.forward(x)
        x=self.average_pooling1.forward(x)
        
        x=self.Convolution2.forward(x)
        x=self.tanh_act2.forward(x)
        x=self.average_pooling2.forward(x)
        x=self.flatten.forward(x)
        x=self.Linear_layer1.forward(x)
        x=self.tanh_act3.forward(x)
        x=self.Linear_layer2.forward(x)
        x=self.softmax.forward(x)
        return x
        
    def loss(self,y_pred,y_actual):
        return self.loss_.forward(y_actual,y_pred)
    
    def backward(self,y_actual):
        
        dx=self.softmax.backward(y_actual)
        dx=self.Linear_layer2.backward(dx)
        dx=self.tanh_act3.backward(dx)
        dx=self.Linear_layer1.backward(dx)
        dx=self.flatten.backward(dx)
        dx=self.average_pooling2.backward(dx)
        dx=self.tanh_act2.backward(dx)
        dx=self.Convolution2.backward(dx)
        dx=self.average_pooling1.backward(dx)
        dx=self.tanh_act1.backward(dx)
        dx=self.Convolution1.backward(dx)
        return dx
        
        
    def update_parameters(self):
        self.Convolution1.weights, self.Convolution1.bias = \
            self.optimizer.update(
                self.Convolution1.weights,
                self.Convolution1.dw_weights,
                self.Convolution1.bias,
                self.Convolution1.dw_bias
            )

        self.Convolution2.weights, self.Convolution2.bias = \
            self.optimizer.update(
                self.Convolution2.weights,
                self.Convolution2.dw_weights,
                self.Convolution2.bias,
                self.Convolution2.dw_bias
            )

        self.Linear_layer1.weights, self.Linear_layer1.bias = \
            self.optimizer.update(
                self.Linear_layer1.weight,
                self.Linear_layer1.d_weights,
                self.Linear_layer1.bias,
                self.Linear_layer1.d_bias
            )

        self.Linear_layer2.weights, self.Linear_layer2.bias = \
            self.optimizer.update(
                self.Linear_layer2.weight,
                self.Linear_layer2.d_weights,
                self.Linear_layer2.bias,
                self.Linear_layer2.d_bias
            )