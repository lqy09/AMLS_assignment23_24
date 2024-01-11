from .autoencoder import *
from .utils import *
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import random
import numpy as np

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class PathmnistDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

def preprocessing_feature_extraction(xtrain, ytrain, xval, yval, xtest, ytest):
    seed_everything()
    autoencoder = load_autoencoder("B/Models/conv_autoencoder.pth")
    xtrain_new = reduce_dimension(autoencoder, xtrain)
    xval_new = reduce_dimension(autoencoder, xval)
    xtest_new = reduce_dimension(autoencoder, xtest)
    ytrain_new = ytrain.ravel()
    yval_new = yval.ravel()
    ytest_new = ytest.ravel()
    return xtrain_new, ytrain_new, xval_new, yval_new, xtest_new, ytest_new

def preprocessing_no_extraction(xtrain, ytrain, xval, yval, xtest, ytest):
    n_train, n_val, n_test = len(xtrain), len(xval), len(xtest)
    xtrain_new = xtrain.reshape((n_train,-1))
    xval_new = xval.reshape((n_val,-1))
    xtest_new= xtest.reshape((n_test,-1))
    ytrain_new = ytrain.ravel()
    yval_new = yval.ravel()
    ytest_new = ytest.ravel()
    return xtrain_new, ytrain_new, xval_new, yval_new, xtest_new, ytest_new

def preprocessing_CNN(xtrain, ytrain, xval, yval, xtest, ytest, batch_size=64):
    """
    x: np.ndarray nx28x28x3 -> tensor nx3x28x28
    
    return:
    dataloader
    """
    # normalize in range 0-1
    xtrain_norm = xtrain/255.0
    xval_norm = xval/255.0
    xtest_norm = xtest/255.0

    generator = torch.Generator()
    xtrain_tensor = torch.tensor(xtrain_norm, dtype=torch.float32)
    xtrain_tensor = xtrain_tensor.permute(0, 3, 1, 2)
    ytrain_tensor = torch.tensor(ytrain, dtype=torch.long).squeeze()
    train_dataset = TensorDataset(xtrain_tensor, ytrain_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    xval_tensor = torch.tensor(xval_norm, dtype=torch.float32)
    xval_tensor = xval_tensor.permute(0, 3, 1, 2)
    yval_tensor = torch.tensor(yval, dtype=torch.long).squeeze()
    val_dataset = TensorDataset(xval_tensor, yval_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=generator)

    xtest_tensor = torch.tensor(xtest_norm, dtype=torch.float32)
    xtest_tensor = xtest_tensor.permute(0, 3, 1, 2)
    ytest_tensor = torch.tensor(ytest, dtype=torch.long).squeeze()
    test_dataset = TensorDataset(xtest_tensor, ytest_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=generator)

    return train_dataloader, val_dataloader, test_dataloader

def preprocessing_resnet(xtrain, ytrain, xval, yval, xtest, ytest, batch_size=64):    
    """
    x: np.ndarray nx28x28x3 -> tensor nx3x224x224
    """

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = PathmnistDataset(xtrain, ytrain.squeeze(), transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = PathmnistDataset(xval, yval.squeeze(), transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = PathmnistDataset(xtest, ytest.squeeze(), transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
