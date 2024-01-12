import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

    
class Autoencoder(nn.Module):
    """ architecture adapted from https://www.geeksforgeeks.org/implement-convolutional-autoencoder-in-pytorch-with-cuda/"""
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.Sigmoid()
        )
         
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def train_autoencoder(xtrain, epochs=100, save_model = False):
    """ input a ndarray and train the autoencoder 
    Args:
    xtrain (ndarray): training dataset
    epochs: training epoch
    save_model: whether to save the model in Model directory

    Returns:
    model: the trained autoencoder
    loss_history: the loss during training
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    xtrain_tensor = torch.tensor(xtrain/255.0, dtype=torch.float32)
    xtrain_tensor = xtrain_tensor.permute(0, 3, 1, 2)
    xtrain_dataset = TensorDataset(xtrain_tensor, xtrain_tensor)
    xtrain_data_loader = DataLoader(xtrain_dataset, batch_size=128, shuffle=True)
    # Train the autoencoder
    num_epochs = epochs
    loss_history = []
    for epoch in range(num_epochs):
        for data in xtrain_data_loader:
            img, _ = data
            img = img.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
        if epoch % 5== 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        loss_history.append(loss.item())
    if save_model:
        # Save the model
        torch.save(model.state_dict(), 'B/Models/conv_autoencoder.pth')
    return model, loss_history

def visualize_reconstruction(model, xtrain):
    """ visualize the reconstruction result of model on a few examples of xtrain"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        xtrain_tensor = torch.tensor(xtrain/255.0, dtype=torch.float32)
        xtrain_tensor = xtrain_tensor.permute(0, 3, 1, 2).to(device)
        recon = model(xtrain_tensor)
    plt.figure(dpi=250)
    fig, ax = plt.subplots(2, 7, figsize=(15, 4))
    for i in range(7):
        ax[0, i].imshow(xtrain_tensor[i].cpu().numpy().transpose((1, 2, 0)))
        ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
        ax[0, i].axis('OFF')
        ax[1, i].axis('OFF')
    plt.savefig('B/figures/reconstruction.png')

def load_autoencoder(path):
    """ input a string of the path to the autoencoder and returns the model """
    model = Autoencoder()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(path))
    return model

def reduce_dimension(model, x):
    """ returns the encoded features of x """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_tensor = torch.tensor(x/255.0, dtype=torch.float32)
    x_tensor = x_tensor.permute(0, 3, 1, 2)
    with torch.no_grad():
        x_new = model.encoder(x_tensor.to(device))
        n_sample = len(x)
        # flatten
        x_encoded = x_new.cpu().numpy().reshape(n_sample, -1)
    return x_encoded