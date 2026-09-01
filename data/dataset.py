import numpy as np
from torchvision import datasets
from pathlib import Path

def load_mnsit(data_dir="/Users/lalitramanmishra/lenet5-from-scratch/data",download=False):
    
    train_dataset=datasets.MNIST(
        root=data_dir,
        download=download,
        train=True
    )
    
    test_dataset=datasets.MNIST(
        root=data_dir,
        train=False,
        download=download
    )
    
    x_train=np.array(train_dataset.data)
    y_train=np.array(train_dataset.targets)
    
    x_test=np.array(test_dataset.data)
    y_test=np.array(test_dataset.targets)
    
    return x_train,y_train,x_test,y_test

def preprocess_image(img):
    "This converts the 28 by 28 image to standard 32 by 32 image"
    # padded_img= np.zeros(32,32)
    # padded_img[2:30,2:30]=img
    
    padded_img=np.pad(img,pad_width=((0,0),(2,2),(2,2)),mode="constant",constant_values=0)
    
    normalized_img=padded_img/255
    
    X=normalized_img[:,np.newaxis, : , :]
    return X
    

def prepare_MNIST(data_dir="/Users/lalitramanmishra/lenet5-from-scratch/data"):
    data_path=Path(data_dir) / "MNIST"
    download_needed = not data_path.exists() or not any(data_path.iterdir())
    X_train, y_train, X_test, y_test = load_mnsit(data_dir, download=download_needed)
        
    X_train=preprocess_image(X_train)
    X_test=preprocess_image(X_test)
    return X_train,y_train,X_test,y_test
    
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_MNIST()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    
    