import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, input_channels=3, output_size=9, 
                 conv_layers=[(16, 3, 1, 1), (32, 3, 1, 1)], 
                 fc_layers=[32*7*7, 128], 
                 pool_size=2, activation_func=F.relu, dropout_conv=0.0, dropout_fc=0.0):
        """
        A CNN network with configurable hyperparameters

        Args:
        - input_channels (int): channel num of input image
        - output_size (int): output num
        - conv_layers (list of tuples): a list of conv layer configs, each element (num_filters, kernel_size, stride, padding)
        - fc_layers (list of int): a list of fc layer configs, each element unit, must match the output of conv layers
        - pool_size (int): the size of pooling kernel
        - activation_func : the activation function chosen from (F.relu, torch.sigmoid, torch.tanh, etc.)
        - dropout_conv : the dropout ratio of convolutional layer
        - dropout_fc : the dropout ratio of fully connected layer
        """
        super(CNN, self).__init__()
        self.pool_size = pool_size
        self.activation_func = activation_func
        
        self.conv_layers = nn.ModuleList()
        self.conv_dropout_layers = nn.ModuleList()
        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            if i == 0:
                in_channels = input_channels
            else:
                in_channels = conv_layers[i-1][0]
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            self.conv_dropout_layers.append(nn.Dropout2d(dropout_conv))

        self.fc_layers = nn.ModuleList()
        self.fc_dropout_layers = nn.ModuleList()
        for i, _ in enumerate(fc_layers):
            if i > 0:
                # must match conv layer output
                self.fc_layers.append(nn.Linear(fc_layers[i-1], fc_layers[i]))
                self.fc_dropout_layers.append(nn.Dropout(dropout_fc))
        self.output_layer = nn.Linear(fc_layers[-1], output_size)


    def forward(self, x):
        for conv_layer, dropout_layer in zip(self.conv_layers, self.conv_dropout_layers):
            x = dropout_layer(self.activation_func(F.max_pool2d(conv_layer(x), self.pool_size)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        for fc_layer, dropout_layer in zip(self.fc_layers, self.fc_dropout_layers):
            x = dropout_layer(self.activation_func(fc_layer(x)))
        x = self.output_layer(x)
        return x
    
def train_cnn(cnn, train_dataloader, val_dataloader=None, epochs=20, lr=0.001, use_validation=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn.to(device)
    epochs = epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    training_loss, val_accuracy = [], []
    for epoch in range(epochs):
        running_loss = 0.0
        cnn.train()
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            output = cnn(images) # forward
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        training_loss.append(running_loss/len(train_dataloader))
        print(f'Epoch {epoch+1}, Training Loss: {running_loss/len(train_dataloader)}')
        if use_validation:
            acc, _ = eval_cnn(cnn, val_dataloader)
            val_accuracy.append(acc)
            print(f'Epoch {epoch+1}, Validated Accuracy: {acc}')
    print("training complete")
    return training_loss, val_accuracy

def eval_cnn(cnn, val_dataloader):
    cnn.eval()
    correct = 0
    total = 0
    all_probs = np.zeros((0, 9))
    all_labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            output = cnn(images)
            _, predict = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
            all_probs = np.vstack([all_probs, output.cpu().detach().numpy()])
            all_labels.extend(labels.cpu().numpy())
        accuracy = correct/total
        one_hot_labels = np.eye(9)[all_labels]
        auc = np.mean([roc_auc_score(one_hot_labels[:, i], all_probs[:, i]) for i in range(9)])
    return accuracy, auc

def lr_tuning(train_dataloader, val_dataloader, lrs=[0.01, 0.003, 0.001, 0.0003, 0.0001]):
    fig, ax = plt.subplots(2, 1, figsize=(10,8))
    for lr in lrs:
        conv = [(16, 3, 1, 1), (32, 3, 1, 1)]
        fc = [32*7*7, 128]
        cnn = CNN(input_channels=3, output_size=9, conv_layers=conv, fc_layers=fc, dropout_conv=0.2, dropout_fc=0.5)
        # print(cnn)
        training_loss, val_accuracy = train_cnn(cnn, train_dataloader, val_dataloader=val_dataloader, epochs=75, lr = lr, use_validation=True)
        ax[0].plot(training_loss, label=f'lr={lr}')
        ax[0].set_xlabel("epoches")
        ax[0].set_ylabel("training loss value")
        ax[0].set_title("training loss")
        ax[0].legend()
        ax[0].grid(ls=':')
        ax[1].plot(val_accuracy, label=f'lr={lr}')
        ax[1].legend()
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("validation accuracy value")
        ax[1].set_title("validation accuracy")
        ax[1].grid(ls=':')

    plt.tight_layout()
    plt.savefig("B/figures/lr_tuning.png")

def conv_layer_tuning(train_dataloader, 
                      val_dataloader,
                      convs = [[(8, 3, 1, 1), [16, 3, 1, 1]],
                               [(16, 3, 1, 1), (32, 3, 1, 1)], 
                               [(32, 3, 1, 1), (64, 3, 1, 1)], 
                               [(16, 3, 1, 1), (32, 3, 1, 1), (64, 3, 1, 1)], 
                               [(32, 3, 1, 1), (64, 3, 1, 1), (128, 3, 1, 1)]],
                      fcs = [[16*7*7, 128],
                             [32*7*7, 128], 
                             [64*7*7, 128],
                             [64*3*3, 128],
                             [128*3*3, 128]],
                      filters=["8,16", "16,32", "32,64", "16,32,64", "32,64,128"]):
                      
    fig, ax = plt.subplots(2, 1, figsize=(10,8))
    for conv, fc, filter in zip(convs, fcs, filters):
        cnn = CNN(input_channels=3, output_size=9, conv_layers=conv, fc_layers=fc, dropout_conv=0.2, dropout_fc=0.5)
        training_loss, val_accuracy = train_cnn(cnn, train_dataloader, val_dataloader=val_dataloader, epochs=75, lr = 0.001, use_validation=True)
        ax[0].plot(training_loss, label=f'num filters={filter}')
        ax[0].set_xlabel("epoches")
        ax[0].set_ylabel("training loss value")
        ax[0].set_title("training loss")
        ax[0].legend()
        ax[0].grid(ls=':')
        ax[1].plot(val_accuracy, label=f'num filters={filter}')
        ax[1].legend()
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("validation accuracy value")
        ax[1].set_title("validation accuracy")
        ax[1].grid(ls=':')

    plt.tight_layout()
    plt.savefig("B/figures/conv_tuning.png")

def fc_layer_tuning(train_dataloader, 
                    val_dataloader,
                    fcs=[[128*3*3, 64],
                        [128*3*3, 128],
                        [128*3*3, 256],
                        [128*3*3, 256, 128],
                        [128*3*3, 256, 128, 64]],
                    fclayers=["64", "128", "256", "256,128", "256,128,64"]):
    fig, ax = plt.subplots(2, 1, figsize=(10,8))
    for fc, fclayers in zip(fcs, fclayers):
        conv = [(32, 3, 1, 1), (64, 3, 1, 1), (128, 3, 1, 1)]
        cnn = CNN(input_channels=3, output_size=9, conv_layers=conv, fc_layers=fc, dropout_conv=0.2, dropout_fc=0.5)
        # print(cnn)
        training_loss, val_accuracy = train_cnn(cnn, train_dataloader, val_dataloader=val_dataloader, epochs=75, lr = 0.001, use_validation=True)
        ax[0].plot(training_loss, label=f'fc_layer_units={fclayers}')
        ax[0].set_xlabel("epoches")
        ax[0].set_ylabel("training loss value")
        ax[0].set_title("training loss")
        ax[0].legend()
        ax[0].grid(ls=':')
        ax[1].plot(val_accuracy, label=f'fc_layer_units={fclayers}')
        ax[1].legend()
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("validation accuracy value")
        ax[1].set_title("validation accuracy")
        ax[1].grid(ls=':')

    plt.tight_layout()
    plt.savefig("B/figures/fc_tuning.png")
    
