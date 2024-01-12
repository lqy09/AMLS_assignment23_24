# ELEC0134 AMLS Assignment

## Introduction

This assignment consists of two different tasks on MedMNIST dataset. A is a binary classification problem for PneumoniaMNIST and B is a multiclass classification problem for PathMNIST.

## Task A

- Trained CNN models are saved in **Model** folder
- Figures in the report are saved in **figure** folder
- `preprocessing.py` contains functions to perform data augmentation and preprocessing
- `knn.py` is used to perform hyperparameter tuning for KNN model
- `svm.py` is used to perform hyperparameter tuning for SVM model
- `cnn.py` implements a custom CNN model and all training, evaluation and hyperparameter tuning functions

## Task B
- Trained autoencoder, cnn, resnet models are saved in **Model** folder
- Figures in the report are saved in **figure** folder
- `preprocessing.py` contains functions to perform feature extraction and preprocessing
- `randomforests.py` is used to perform hyperparameter tuning for random forest model
- `cnn.py` implements a custom CNN model and all training, evaluation and hyperparameter tuning functions
- `resnet.py` modifies pretrained ResNet-18 and performs hyperparameter tuning to achieve better performance

## Requirements

- python=3.8

The `requirements.txt` file is in directory A. The required packages are as follows.

- notebook
- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn
- torch==1.12.0+cu116
- torchvision==0.13.0+cu116
- torchaudio==0.12.0

To install requirements please run
```bash
pip install -r A/requirements.txt
```

## Quick Start

To reproduce the results in the report and hyperparameter tuning process, run
```bash
python main.py
```
To skip long runtime, modify the arguments in main.py file.