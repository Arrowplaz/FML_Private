import pandas as pd
import torch
import torch.nn as nn
from process import prepare_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from copy import deepcopy as dc

X, y = prepare_data(0, 7)


print(X)


print('***SHAPE***')
print(X.shape, y.shape)
# (...,7) (...)
# 7 time steps back
# x is the 7 previous values
# y is the "current" value
# print(X[100])
# print(y[100])
# [-0.45990756 -0.43743811 -0.41579209 -0.3940857  -0.37252854 -0.35360555
#  -0.33486112]
# -0.4843011665370893
train_cu = int(X.shape[0]*0.6)

X_train = X[:train_cu]
print(X_train)
y_train = y[:train_cu]
X_test = X[train_cu:]
print(X_test)
y_test = y[train_cu:]

X_train = torch.tensor(X_train).unsqueeze(1).permute(0,2,1).float()
y_train = torch.tensor(y_train).unsqueeze(1).float()
X_test = torch.tensor(X_test).unsqueeze(1).permute(0,2,1).float()
y_test = torch.tensor(y_test).unsqueeze(1).float()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

from torch.utils.data import DataLoader

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# for _, batch in enumerate(train_loader):
#      x_batch, y_batch = batch[0].to(device), batch[1].to(device)
#      print(x_batch.shape, y_batch.shape)
#      break


#####################################################################################
# This was coded up just to help us understand how the inputs were passing through the LSTM
# testing the effect of different numbers of layers was done using the nn.LSTM module
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #input gate weights
        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        #forget gate weights
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
        #hidden state weights
        self.U_h = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))
        
        #output gate weights
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # no weights for cell state! (vanishing/exploding gradient)
        
        self.init_weights()
        self.fc = nn.Linear(hidden_size, 1)

                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, init_states=None):
        
        """
        assumes x.shape represents (batch_size, sequence_size, input_size)
        """
        batch_size, seq_size, _ = x.size()
        hidden_seq = []
        
        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch_size, self.hidden_size).to(x.device),
                torch.zeros(batch_size, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states
            
        for t in range(seq_size):
            x_t = x[:, t, :]
            
            i_t = torch.sigmoid(torch.matmul(x_t, self.U_i) + torch.matmul(h_t, self.V_i) + self.b_i)
            f_t = torch.sigmoid(torch.matmul(x_t, self.U_f) + torch.matmul(h_t, self.V_f) + self.b_f)
            g_t = torch.tanh(torch.matmul(x_t, self.U_h) + torch.matmul(h_t, self.V_h) + self.b_h)
            o_t = torch.sigmoid(torch.matmul(x_t, self.U_o) + torch.matmul(h_t, self.V_o) + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            
            hidden_seq.append(h_t.unsqueeze(0))
        
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        out = self.fc(hidden_seq[:,-1,:])
        return out
#####################################################################################


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(1, 64, 3)
# model = CustomLSTM(1,64)
model.to(device)


learning_rate = 0.001
num_epochs = 20
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_one_epoch(epoch):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:# print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1, avg_loss_across_batches))
    running_loss = 0.0
    print()

def accuracy():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()

## train
for epoch in range(num_epochs):
    train_one_epoch(epoch)
    accuracy()

## testing (IIS)
with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()


print("Y_train: ", y_train.shape)
print("predicted: ", predicted.shape)
plt.plot(y_train, label='Actual Value')
plt.plot(predicted, label='Predicted Value')
plt.xlabel('Day')
plt.ylabel('Value')
plt.legend()
plt.show()


## testing (OOS)
with torch.no_grad():
    predicted = model(X_test.to(device)).to('cpu').numpy()

plt.plot(y_test, label='Actual Close')
plt.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()


## Autoregressive forecasting

# Initialize an empty list to store the predicted values
predicted_values = []

# Predict future values iteratively using autoregressive forecasting
with torch.no_grad():
    # Initialize the input sequence with the first 7 real data points from the test set
    # print("X_test: ", X_test.shape)
    print("X_test: ", X_test[0].shape)
    print("X_test: ", X_test[1])
    print("X_test: ", X_test[2])
    print("X_test: ", X_test[3])
    print("X_test: ", X_test[4])
    print("X_test: ", X_test[5])
    input_sequence = X_test[0].unsqueeze(0).to(device) 

    # Define the number of future steps to predict
    future_steps = len(y_test)

    x = 2

    for _ in range(future_steps):
        # Predict the next data point using the model
        predicted_value = model(input_sequence)
        #add a random number between -0.05 and 0.05 to the predicted value every 4 times

        #add some noise decay
        if _ % x == 0:
            #if there has been a positive trend between the last 2 days give some positive noise
            #check if the predicted values exist
            if len(predicted_values) > 2 and predicted_value.item() > predicted_values[-1] :
                predicted_value += np.random.uniform(0.0025, 0.005)
                print('positive noise')
            #if there has been a negative trend between the last 2 days give some negative noise
            elif len(predicted_values) > 2 and predicted_value.item() < predicted_values[-1]:
                predicted_value -= np.random.uniform(0.0025, 0.005)
                print('negative noise')
            x = 2*x
           

        # Append the predicted value to the list
        predicted_values.append(predicted_value.item())

        # Update the input sequence by removing the first data point and appending the predicted value
        input_sequence = torch.cat([input_sequence[:, 1:], predicted_value.unsqueeze(-1)], dim=1)  # Modify dimensions

# Calculate the mean squared error
mse = mean_squared_error(y_test, predicted_values)
print('Mean Squared Error:', mse)


# Plot the actual and predicted values
plt.plot(y_test, label='Actual Value')
plt.plot(predicted_values, label='Predicted Value')
plt.xlabel('Day')
plt.ylabel('Value')
plt.legend()
plt.show()



# with torch.no_grad():
#     predicted = model(X_test.to(device)).to('cpu').numpy()

# plt.plot(y_test, label='Actual Value')
# plt.plot(predicted, label='Predicted Value')
# plt.xlabel('Day')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
