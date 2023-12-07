import pandas as pd
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv
import sklearn.metrics as skm
import sklearn.model_selection as skms
import torch.utils.data as tud

#loading saved tensor from data 
train_features = torch.load(r'C:\Visual Studio Coding\pytorch\linear regression\Charges_Prediction\data\train_features.pt')
train_labels = torch.load(r'C:\Visual Studio Coding\pytorch\linear regression\Charges_Prediction\data\train_labels.pt')
test_features = torch.load(r'C:\Visual Studio Coding\pytorch\linear regression\Charges_Prediction\data\test_features.pt')
test_labels = torch.load(r'C:\Visual Studio Coding\pytorch\linear regression\Charges_Prediction\data\test_labels.pt')

#create torch dataset, takes two torch tensors, the features tensor and the labels tensor 
train_data = tud.TensorDataset(train_features, train_labels)
test_data = tud.TensorDataset(test_features, test_labels)


#taking tensor datasets and feeding them to data loader, this returns a iterable of batches of data
batch_size = 128
train_loader = tud.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = tud.DataLoader(test_data, batch_size=batch_size)

#setting up the model 
n_samples, n_features = train_features.shape
x_para = n_features
y_para = 1 
model = nn.Linear(x_para, y_para)

#setting up the loss and optimizer 
loss_func = nn.MSELoss()

#learning rate not needed for Adam, RMSprop
optimizer = torch.optim.Adam(model.parameters())

#training loop
epochs = 1001
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

torch.save(model, 'data/model.pth')

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