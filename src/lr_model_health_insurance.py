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

#inputs two features tensor and labels tensor, test_size determines the percentage of the data to use for test, random_state is the seed of the random spliting 
train_features, test_features, train_labels, test_labels = skms.train_test_split(features_tensor, labels_tensor, test_size=0.2, random_state=42)

#create torch dataset, takes two torch tensors, the features tensor and the labels tensor 
train_data = tud.TensorDataset(train_features, train_labels)
test_data = tud.TensorDataset(test_features, test_labels)

#taking tensor datasets and feeding them to data loader, this returns a iterable of batches of data
batch_size = 64
train_loader = tud.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = tud.DataLoader(test_data, batch_size=batch_size)


#setting up the model 
n_samples, n_features = features.shape
x_para = n_features
y_para = 1 
model = nn.Linear(x_para, y_para)

#setting up the loss and optimizer 
loss_func = nn.MSELoss()

#learning rate not needed for Adam, RMSprop
optimizer = torch.optim.RMSprop(model.parameters())

#training loop
epochs = 10000
for epoch in range(epochs):
    #set model into training mode 
    model.train()
    #setting running loss
    running_loss = 0 
    for features, labels in train_loader:
        #clear optimizer
        optimizer.zero_grad()
        #get y predicted
        predicted_labels = model(features)
        #get loss 
        loss = loss_func(predicted_labels, labels.view(-1, 1))
        #calculated direction of gradient
        loss.backward()
        #get the new theta
        optimizer.step()
        #multipe the loss by the size of the batch to scale it to make the loss 
        #representative of the loss across the entire batch (accounts for batch of different sizes)
        running_loss += loss.item() * features.size(0)
    loss = running_loss / len(train_loader.dataset)

    if epoch % 1000 == 0:
        print(f"Training Loss: {loss}")
    
model.eval()
test_loss_cul = 0
r_squared_total = 0 
with torch.no_grad():
    for features, labels in test_loader:
        #calculating predicted 
        predicted_labels = model(features)

        labels = labels.view(-1)
        predicted_labels = predicted_labels.view(-1)

        loss = loss_func(predicted_labels, labels)
        test_loss_cul += loss.item() * features.size(0) 

        #calculating the r squared score 
        r_squared_score = skm.r2_score(labels, predicted_labels)
        r_squared_total += r_squared_score

    test_loss = test_loss_cul / len(test_loader.dataset)
    average_r_squared = r_squared_total / len(test_loader)
    
    print(f"Test Loss: {test_loss}")
    print(f"R2 Value: {average_r_squared}")


    # # #casting the list to a float tensor to use 
    # # test = torch.FloatTensor([23, 0, 22.8, 0, 0, 0])
    # # print(f"The test case of 18 = {model(test) * 1000}")