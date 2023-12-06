import pandas as pd
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv
import torchmetrics.functional as tm



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
    standard_X = (X - feature_mean) / feature_std
    return standard_X

#set up the data 
data = pd.read_csv('C:\Visual Studio Coding\pytorch\linear regression\data\insurance.csv')
categories = ['sex', 'smoker', 'region']
data = pd.get_dummies(data, columns=categories)

training_data = data.drop('charges', axis=1).values
labels = data['charges'].values



# #converting the data to numeric 
# #male = 0, female = 1
# x_data[x_data == 'male'] = 0
# x_data[x_data == 'female'] = 1

# #non-smoker = 0, smoker = 1, 
# x_data[x_data == 'no'] = 0
# x_data[x_data == 'yes'] = 1

# #northeast = 0, southeast = 1, northwest = 2, southwest = 3
# x_data[x_data == 'northeast'] = 0
# x_data[x_data == 'southeast'] = 1
# x_data[x_data == 'northwest'] = 2
# x_data[x_data == 'southwest'] = 3


# #standardizing the data 
# x_numpy = standardize(x_data.astype(np.float32))
# y_numpy = standardize(y_data.astype(np.float32))


# #converting the data to torch tensor 
# x = torch.from_numpy(x_numpy.astype(np.float32)) 
# y = torch.from_numpy(y_numpy.astype(np.float32)) 

# #setting up the model 
# n_samples, n_features = x.shape
# x_para = n_features
# y_para = 1 
# lr_model = nn.Linear(x_para, y_para)

# #setting up the loss and optimizer 
# loss_func = nn.MSELoss()

# #learning rate not needed for Adam, RMSprop
# optimizer = torch.optim.Adam(lr_model.parameters())

# #training loop
# epochs = 10000
# for epoch in range(epochs):
#     #get y predicted
#     y_predicted = lr_model(x)
#     #get loss 
#     loss = loss_func(y_predicted, y)
#     #calculated direction of gradient
#     loss.backward()
#     #get the new theta
#     optimizer.step()
#     #clear optimizer
#     optimizer.zero_grad()

#     if epoch % 100 == 0:
#         print(loss.item())

# print('here')

# #getting regression line 
# theta = lr_model(x).detach().numpy()

# #calculating the r squared score 
# r_squared_score = tm.r2_score(theta, y)
# print(f"The R-squared score is {r_squared_score}")

# #casting the list to a float tensor to use 
# test = torch.FloatTensor([23, 0, 22.8, 0, 0, 0])
# print(f"The test case of 18 = {lr_model(test) * 1000}")