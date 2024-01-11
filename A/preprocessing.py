import numpy as np
import random
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def training_data_augmentation(xtrain, ytrain, num_transformed_images=2280):
    """ augment data with label 0 to make it a balanced dataset """

    xtrain_label0 = xtrain[ytrain.flatten()==0]
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ToTensor(),
    ])
    transformed_images = []
    while len(transformed_images) < num_transformed_images:
        for img in xtrain_label0:
            img_tensor = torch.tensor(img, dtype=torch.uint8)
            img_transformed = transform(img_tensor).numpy() * 255
            transformed_images.append(img_transformed)
            if len(transformed_images) >= num_transformed_images:
                break
    augmented_images = np.array(transformed_images)[:,0,:,:]
    augmented_xtrain = np.concatenate([xtrain, augmented_images], axis=0)
    augmented_ytrain = np.concatenate([ytrain, np.zeros((2280, 1))])
    return augmented_xtrain, augmented_ytrain

def preprocessing_CNN(xtrain, ytrain, xval, yval, xtest, ytest, batch_size=64):
    """
    x: np.ndarray nx28x28 -> tensor nx1x28x28
    
    return:
    dataloader
    """
    # normalize in range 0-1
    xtrain_norm = xtrain/255.0
    xval_norm = xval/255.0
    xtest_norm = xtest/255.0

    generator = torch.Generator()
    xtrain_tensor = torch.tensor(xtrain_norm, dtype=torch.float32).unsqueeze(1) # add a channel
    ytrain_tensor = torch.tensor(ytrain, dtype=torch.long).squeeze()
    train_dataset = TensorDataset(xtrain_tensor, ytrain_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    xval_tensor = torch.tensor(xval_norm, dtype=torch.float32).unsqueeze(1) # add a channel
    yval_tensor = torch.tensor(yval, dtype=torch.long).squeeze()
    val_dataset = TensorDataset(xval_tensor, yval_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    xtest_tensor = torch.tensor(xtest_norm, dtype=torch.float32).unsqueeze(1) # add a channel
    ytest_tensor = torch.tensor(ytest, dtype=torch.long).squeeze()
    test_dataset = TensorDataset(xtest_tensor, ytest_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    return train_dataloader, val_dataloader, test_dataloader

def preprocessing_KNN_SVM(xtrain, ytrain, xval, yval, xtest, ytest):
    """
    preprocessing for knn and svm

    Args:
    - xtrain, xval, xtest (ndarray): nx28x28 -> nx784
    - ytrain, yval, ytest (ndarray): nx1 -> n,

    Returns:
    Flattened and shuffled ndarray
    """
    # normalize in range 0-1
    xtrain_norm = xtrain/255.0
    xval_norm = xval/255.0
    xtest_norm = xtest/255.0
    
    n_train, n_val, n_test = len(xtrain), len(xval), len(xtest)
    xtrain_new = xtrain_norm.reshape((n_train,-1))
    xval_new = xval_norm.reshape((n_val,-1))
    xtest_new= xtest_norm.reshape((n_test,-1))
    ytrain_new = ytrain.ravel()
    yval_new = yval.ravel()
    ytest_new = ytest.ravel()

    # shuffle xtrain and ytrain
    idx = np.random.permutation(len(xtrain_new))
    xtrain_new = xtrain_new[idx]
    ytrain_new = ytrain_new[idx]

    return xtrain_new, ytrain_new, xval_new, yval_new, xtest_new, ytest_new
