import pandas as pd
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv
import sklearn.metrics as skm
import sklearn.model_selection as skms
import torch.utils.data as tud

def standardize (X):
    """
    X is a 2D Numpy array that each row is a feature vector of a data point (columns are features)
    Returns a standardized 2D Numpy array 
    """
    #cacluting mean along the first axis 
    feature_mean = X.mean(0)
    #calculating the mean along the first axis 
    feature_std = X.std(0)
    #standardize the features
    return (X - feature_mean) / feature_std
    
#set up the data 
data = pd.read_csv('C:\Visual Studio Coding\pytorch\linear regression\Charges_Prediction\data\insurance.csv')
#getting categoires to convert to numeric values 
categories = ['sex', 'smoker', 'region']
#converting these pd collumns to numeric
data = pd.get_dummies(data, columns=categories)

#getting training data and labels .values returns a numpy array, using astype(float) converts all true to 1 and all false to 0 
features = standardize(data.drop('charges', axis=1).values.astype(float))
labels = standardize(data['charges'].values.astype(float))

#features 
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

#inputs two tensors, features tensor and labels tensor, test_size determines the percentage of the data to use for test, random_state is the seed of the random spliting 
#returns tensors 
train_features, test_features, train_labels, test_labels = skms.train_test_split(features_tensor, labels_tensor, test_size=0.2, random_state=42)

torch.save(train_features, 'data/train_features.pt')
torch.save(test_features, 'data/test_features.pt')
torch.save(train_labels, 'data/train_labels.pt')
torch.save(test_labels, 'data/test_labels.pt')



