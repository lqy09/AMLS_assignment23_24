from .preprocessing import *
from .randomforests import *
from .cnn import *
from .resnet import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
from torchvision import transforms
from sklearn.metrics import accuracy_score

def train_all_models(train_randomforest=True, train_cnn=True, train_resnet=True):
    """ training all fine-tuned models for task B, if one process is long, just turn the arguments to False"""
    seed_everything(seed=42)
    print("loading data...")
    data = np.load('./Datasets/pathmnist.npz')
    xtrain, ytrain, xval, yval, xtest, ytest = data["train_images"], data["train_labels"], data["val_images"], data["val_labels"], data["test_images"], data["test_labels"]
    data.close()
    print("loading data complete...")

    if train_randomforest:
        print("train fine-tuned Random Forest model with autoencoder preprocessing...")
        xtrain_new, ytrain_new, xval_new, yval_new, xtest_new, ytest_new = preprocessing_feature_extraction(xtrain, ytrain, xval, yval, xtest, ytest)
        forest = RandomForestClassifier(n_estimators=200, max_features='sqrt', criterion='gini')
        forest.fit(xtrain_new, ytrain_new)
        val_acc_forest = accuracy_score(yval_new, forest.predict(xval_new))
        test_acc_forest = accuracy_score(ytest_new, forest.predict(xtest_new))
        print(f"val acc of random forest with feature extraction: {val_acc_forest}, test acc: {test_acc_forest}")

        print("without autoencoder preprocessing...")
        xtrain_new, ytrain_new, xval_new, yval_new, xtest_new, ytest_new = preprocessing_no_extraction(xtrain, ytrain, xval, yval, xtest, ytest)
        forest = RandomForestClassifier(n_estimators=200, max_features='sqrt', criterion='gini')
        forest.fit(xtrain_new, ytrain_new)
        val_acc_forest = accuracy_score(yval_new, forest.predict(xval_new))
        test_acc_forest = accuracy_score(ytest_new, forest.predict(xtest_new))
        print(f"val acc of random forest without feature extraction: {val_acc_forest}, test acc: {test_acc_forest}")

    if train_cnn:
        print("load fine-tuned CNN model...")
        train_dataloader, val_dataloader, test_dataloader = preprocessing_CNN(xtrain, ytrain, xval, yval, xtest, ytest)
        conv = [(32, 3, 1, 1), (64, 3, 1, 1), (128, 3, 1, 1)]
        fc = [128*3*3, 128]
        cnn = CNN(input_channels=3, output_size=9, conv_layers=conv, fc_layers=fc, dropout_conv=0.2, dropout_fc=0.5)
        cnn.load_state_dict(torch.load("B/Models/bestcnn.pth"))
        val_acc_cnn, val_auc_cnn = eval_cnn(cnn, val_dataloader)
        test_acc_cnn, test_auc_cnn = eval_cnn(cnn, test_dataloader)
        print(f"val ACC of CNN: {val_acc_cnn}, test ACC of CNN: {test_acc_cnn}")
        print(f"val AUC of CNN: {val_auc_cnn}, test AUC of CNN: {test_auc_cnn}")

    if train_resnet:
        print("load fine-tuned ResNet model...")
        train_dataloader, val_dataloader, test_dataloader = preprocessing_resnet(xtrain, ytrain, xval, yval, xtest, ytest)
        resnet = ResNetModel(num_classes=9, unfreeze_layers=['layer3', 'layer4'], pretrained=True)
        resnet.load_state_dict(torch.load("B/Models/bestresnet.pth"))
        val_acc_resnet = eval_resnet(resnet, val_dataloader)
        test_acc_resnet = eval_resnet(resnet, test_dataloader)
        print(f"val acc of resnet: {val_acc_resnet}, test acc: {test_acc_resnet}")

def show_tuning_process(show_forest=True, show_cnn=True, show_resnet=True):
    """ Show the hyperparameter tuning process for task B in the report. If one process is long, just turn the argument to False """
    seed_everything(seed=42)
    print("loading data...")
    data = np.load('./Datasets/pathmnist.npz')
    xtrain, ytrain, xval, yval, xtest, ytest = data["train_images"], data["train_labels"], data["val_images"], data["val_labels"], data["test_images"], data["test_labels"]
    data.close()
    print("loading data complete")
    xtrain_new, ytrain_new, xval_new, yval_new, xtest_new, ytest_new = preprocessing_feature_extraction(xtrain, ytrain, xval, yval, xtest, ytest)

    if show_forest:
        print("cross validation of random forest may take a few hours...")
        param_grid = {
            'n_estimators':[100, 200],
            'criterion':['gini', 'entropy', 'log_loss'],
            'max_features': ['sqrt', 'log2']
        }
        best_classifier, best_params, cv_results = randomforest_tuning(param_grid, xtrain_new, ytrain_new)
        print("best parameter: ", best_params)
    
    if show_cnn:
        print("CNN hyperparameter tuning make take a long time")
        train_dataloader, val_dataloader, test_dataloader = preprocessing_CNN(xtrain, ytrain, xval, yval, xtest, ytest)
        print("start learning rate tuning...")
        lr_tuning(train_dataloader, val_dataloader)
        print("saved plot in B/figures/lr_tuning_aug.png")
        print("start convolutional layer tuning...")
        conv_layer_tuning(train_dataloader, val_dataloader)
        print("saved plot in B/figures/conv_tuning_aug.png")
        print("start fully connected layer tuning...")
        fc_layer_tuning(train_dataloader, val_dataloader)
        print("saved plot in B/figures/fc_tuning_aug.png")

    if show_resnet:
        print("Resnet tuning make take a few hours...")
        train_dataloader, val_dataloader, test_dataloader = preprocessing_resnet(xtrain, ytrain, xval, yval, xtest, ytest)
        tuning_resnet(train_dataloader, val_dataloader)
        print("saved plot in B/figures/resnet_tuning.png")

    print("tuning for task B completed")