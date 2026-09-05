from src.model.lenet5 import LeNet5
from data.dataset import prepare_MNIST
from src.utils import batch_

X_train, y_train, X_test, y_test = prepare_MNIST()


model=LeNet5()
batch=1
i=100
for eochs in range(3):
    print(f"Training Epcoh {eochs+1} started")
    for x_batch,y_batch in batch_(32,X_train,y_train):
        y_pred=model.forward(x_batch)
        loss=model.loss(y_pred,y_batch)
        model.backward(y_batch)
        model.update_parameters()
        if batch%i==0:
            print(f"batches {batch} are completed")
        batch+=1
    print(f"Training Epcoh {eochs} ended")
    
    