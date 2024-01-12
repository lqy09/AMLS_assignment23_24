import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


def ResNetModel(num_classes=9, unfreeze_layers=['layer4'], pretrained=True):
    """
    Args:
    frozen_layers: selected from conv1,bn1,layer1,layer2,layer3,layer4
    """
    model = models.resnet18(pretrained = pretrained)
    for name, child in model.named_children():
        if name in unfreeze_layers:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False
    num_classes = 9
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def train_resnet(model, train_dataloader, val_dataloader, epochs=10, lr=0.001, scheduling=False):
    """ Train a resnet 
    Args:
    - model: resnet model
    - train_dataloader: training dataset
    - val_dataloader: the validation dataset during training
    - epochs: training epochs
    - lr: learning rate
    - scheduling: whether to use learning rate scheduling

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if scheduling:
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    training_loss, val_accuracy = [], []
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images) # forward
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduling:
            scheduler.step()
            print(f"Epoch {epoch+1}: current lr={scheduler.get_last_lr()}")
        training_loss.append(running_loss/len(train_dataloader))
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        val_accuracy.append(val_acc)
        print(f'Epoch {epoch+1}, Training Loss: {training_loss[-1]}, Validation Accuracy: {val_acc}')
    print("training complete")
    return training_loss, val_accuracy

def eval_resnet(model, val_dataloader):
    """ Evaluate resnet model on validation dataloader, returns the validated accuracy"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    return val_acc


def tuning_resnet(train_dataloader, val_dataloader, lrs=[0.001, 0.0001]):
    """ Performs the hyperparameter tuning of ResNet-18"""
    model = ResNetModel()
    fig, ax = plt.subplots(2, 1, figsize=(10,8))        
    for lr in lrs:
        training_loss, val_accuracy = train_resnet(model, train_dataloader, val_dataloader, lr=lr)
        ax[0].plot(training_loss, label=f'lr={lr}')
        ax[1].plot(val_accuracy, label=f'lr={lr}')
    training_loss, val_accuracy = train_resnet(model, train_dataloader, val_dataloader, scheduling=True)
    ax[0].plot(training_loss, label='lr scheduling')
    ax[1].plot(val_accuracy, label='lr scheduling')
    model = ResNetModel(unfreeze_layers=['layer3', 'layer4'])
    training_loss, val_accuracy = train_resnet(model, train_dataloader, val_dataloader, scheduling=True)
    ax[0].plot(training_loss, label='unfreeze layer 3,4')
    ax[1].plot(val_accuracy, label='unfreeze layer 3,4')
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("training loss value")
    ax[0].set_title("training loss")        
    ax[0].legend()
    ax[0].grid(ls=':')
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("validation accuracy value")
    ax[1].set_title("validation accuracy")
    ax[1].legend()
    ax[1].grid(ls=':')
    plt.tight_layout()
    plt.savefig("B/figures/resnet_tuning.png")