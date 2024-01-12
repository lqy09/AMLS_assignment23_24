from .preprocessing import *
from .cnn import *
from .knn import *
from .svm import *
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import torch
from torchvision import transforms
from sklearn.metrics import accuracy_score

def train_all_models():
    seed_everything(seed=42)
    print("loading data...")
    data = np.load('./Datasets/pneumoniamnist.npz')
    xtrain, ytrain, xval, yval, xtest, ytest = data["train_images"], data["train_labels"], data["val_images"], data["val_labels"], data["test_images"], data["test_labels"]
    data.close()
    print("loading data complete...")
    xtrain_aug, ytrain_aug = training_data_augmentation(xtrain, ytrain)
    xtrain_new, ytrain_new, xval_new, yval_new, xtest_new, ytest_new = preprocessing_KNN_SVM(xtrain_aug, ytrain_aug, xval, yval, xtest, ytest)

    print("train fine-tuned KNN model...")
    knn = KNeighborsClassifier(n_neighbors=10, metric='cosine', weights='distance')
    knn.fit(xtrain_new, ytrain_new)
    val_acc_knn = accuracy_score(yval_new, knn.predict(xval_new))
    test_acc_knn = accuracy_score(ytest_new, knn.predict(xtest_new))
    print(f"val acc of KNN: {val_acc_knn}, test acc of KNN: {test_acc_knn}")

    print("train fine-tuned SVM model...")
    svm = SVC(C=3, kernel='rbf', gamma='scale')
    svm.fit(xtrain_new, ytrain_new)
    val_acc_svm = accuracy_score(yval_new, svm.predict(xval_new))
    test_acc_svm = accuracy_score(ytest_new, svm.predict(xtest_new))
    print(f"val acc of SVM: {val_acc_svm}, test acc of SVM: {test_acc_svm}")

    print("load fine-tuned CNN model with data augmentation...")
    train_dataloader, val_dataloader, test_dataloader = preprocessing_CNN(xtrain_aug, ytrain_aug, xval, yval, xtest, ytest)
    conv = [(32, 3, 1, 1), (64, 3, 1, 1), (128, 3, 1, 1)]
    fc = [128*3*3, 128]
    cnn = CNN(input_channels=1, output_size=2, conv_layers=conv, fc_layers=fc, dropout_conv=0.2, dropout_fc=0.5)
    cnn.load_state_dict(torch.load("A/Models/bestcnn_withaug.pth"))
    val_acc_cnn, val_auc_cnn = eval_cnn(cnn, val_dataloader)
    test_acc_cnn, test_auc_cnn = eval_cnn(cnn, test_dataloader)
    print(f"val ACC of CNN: {val_acc_cnn}, test ACC of CNN: {test_acc_cnn}")
    print(f"val AUC of CNN: {val_auc_cnn}, test AUC of CNN: {test_auc_cnn}")

    print("load fine-tuned CNN model without data augmentation...")
    cnn.load_state_dict(torch.load("A/Models/bestcnn.pth"))
    val_acc_cnn, val_auc_cnn = eval_cnn(cnn, val_dataloader)
    test_acc_cnn, test_auc_cnn = eval_cnn(cnn, test_dataloader)
    print(f"val ACC of CNN: {val_acc_cnn}, test ACC of CNN: {test_acc_cnn}")
    print(f"val AUC of CNN: {val_auc_cnn}, test AUC of CNN: {test_auc_cnn}")

def show_tuning_process(show_knn=True, show_svm=True, show_cnn=True):
    seed_everything(seed=42)
    print("loading data...")
    data = np.load('./Datasets/pneumoniamnist.npz')
    xtrain, ytrain, xval, yval, xtest, ytest = data["train_images"], data["train_labels"], data["val_images"], data["val_labels"], data["test_images"], data["test_labels"]
    data.close()
    print("loading data complete")
    xtrain_aug, ytrain_aug = training_data_augmentation(xtrain, ytrain)
    xtrain_new, ytrain_new, xval_new, yval_new, xtest_new, ytest_new = preprocessing_KNN_SVM(xtrain_aug, ytrain_aug, xval, yval, xtest, ytest)

    if show_knn:
        print("plot the tuning of k in KNN in range 1-80...")
        KNN_k_tuning_plot(xtrain_new, ytrain_new)
        print("saved plot in A/figures/KNN_k_tuning_plot.png")
        param_grid = {
            'n_neighbors': [5, 10, 15, 20, 25], 
            'metric': ['cosine', 'manhattan', 'minkowski'],
            'weights': ['uniform', 'distance']
        }
        best_classifier, best_params, cv_results = KNN_tuning(param_grid, xtrain_new, ytrain_new)
        print("best parameter: ", best_params)
    
    if show_svm:
        print("plot the tuning of C in SVM in range 1-50...")
        SVM_C_tuning_plot(xtrain_new, ytrain_new, xval_new, yval_new)
        print("saved plot in A/figures/SVM_C_tuning_plot.png")
        param_grid = {
            'C':[5, 15, 30, 40],
            'gamma':['scale','auto', 0.001, 0.0001],
            'kernel': ['poly', 'rbf', 'sigmoid']
        }
        best_classifier, best_params, cv_results = SVM_tuning(param_grid, xtrain_new, ytrain_new)
        print("best parameter: ", best_params)

    if show_cnn:
        print("start learning rate tuning, it may take a long time...")
        train_dataloader, val_dataloader, test_dataloader = preprocessing_CNN(xtrain_aug, ytrain_aug, xval, yval, xtest, ytest)
        lr_tuning(train_dataloader, val_dataloader)
        print("saved plot in A/figures/lr_tuning_aug.png")
        conv_layer_tuning(train_dataloader, val_dataloader)
        print("saved plot in A/figures/conv_tuning_aug.png")
        fc_layer_tuning(train_dataloader, val_dataloader)
        print("saved plot in A/figures/fc_tuning_aug.png")

    print("tuning for task A completed")