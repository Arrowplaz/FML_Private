import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# File path
file_path = "/Users/ksarna/Downloads/house_prices.csv"

df = pd.read_csv(file_path)

# need dates in some date time format that can be plotted
dates = []
for row,index in df.iterrows():
    for i,date in enumerate(row): 
        if i >= 5:
            dates.append(date)
    break
dates.append('2024-03-31')

iterator = df.iterrows()
next(iterator)# Skip the first row
house_prices = [] # list of lists - each list is sequence of house prices
# train on first 500 regions, test on last 395
# input is first 150 prices, label is next 141
train_prices = []
train_prices_labels = []
test_prices = []
test_prices_labels = []
j = 1
for index, row in iterator:
    total_time_series = []
    last = float(row[0])
    for i, price in enumerate(index):
        if i >= 5:# Skip the first 5 iterations
            price = float(price)
            total_time_series.append(price)
    total_time_series.append(last)
    if j<=500: # training data
        train_prices.append(total_time_series[:150])
        train_prices_labels.append(total_time_series[150:])
    else: # test data
        test_prices.append(total_time_series[:150])
        test_prices_labels.append(total_time_series[150:])
    house_prices.append(total_time_series)
    j += 1



# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))  # Adjust the figure size

# # Plot the data with a blue solid line and circular markers
# plt.plot(dates[60:110], house_prices[0][60:110], marker='o', linestyle='-', color='b', linewidth=2, markersize=8, label='House Prices')

# # Rotate x-axis labels for better readability
# plt.xticks(rotation=45)

# # Add grid lines for better visualization
# plt.grid(True, linestyle='--', alpha=0.5)

# # Add labels and title with larger font sizes
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('Price ($)', fontsize=14)
# plt.title('House Prices (US)', fontsize=16)

# # Add legend
# plt.legend(loc='upper left', fontsize=12)

# # Automatically adjust subplot parameters to give specified padding
# plt.tight_layout()

# # Display the plot
# plt.show()

# https://medium.com/@mrconnor/time-series-forecasting-with-pytorch-predicting-stock-prices-81db0f4348ef

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


# create additional columns for n_steps previous time steps
# now each row represents historical house values
def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Price(t-{i})'] = df['Price'].shift(i)

    df.dropna(inplace=True)

    return df

def prepare_data(column):

    data = {
        'Date': dates,
        'Price': house_prices[column]
    }

    df = pd.DataFrame(data)
    lookback = 7
    shifted_df = prepare_dataframe_for_lstm(df, lookback)
    print(shifted_df)

    shifted_df_as_np = shifted_df.to_numpy()


    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]

    return X, y

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from copy import deepcopy as dc


X = np.array([])
y = np.array([])
for i in range(10):
    X_temp, y_temp = prepare_data(i)
    X_temp = np.array(X_temp)
    y_temp = np.array(y_temp)
    X = np.concatenate((X, X_temp), axis=0) if X.size else X_temp
    y = np.concatenate((y, y_temp), axis=0) if y.size else y_temp




print(X.shape, y.shape)
# (284,7) (284)
# 284 days
# 7 time steps back
# x is the 7 previous values
# y is the "current" value
#print(X[100])
#print(y[100]()
# [-0.45990756 -0.43743811 -0.41579209 -0.3940857  -0.37252854 -0.35360555
#  -0.33486112]
# -0.4843011665370893
train_cu = int(X.shape[0]*0.9)

X_train = X[:train_cu]
y_train = y[:train_cu]
X_test = X[train_cu:]
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

for _, batch in enumerate(train_loader):
     x_batch, y_batch = batch[0].to(device), batch[1].to(device)
     print(x_batch.shape, y_batch.shape)
     break


class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the first LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Define additional LSTM layers
        self.lstms = nn.ModuleList([nn.LSTM(hidden_size, hidden_size, batch_first=True) for _ in range(num_layers - 1)])

        # Define the fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize hidden and cell state for the first layer
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(device)

        # Pass input through the first LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Pass output through additional LSTM layers
        for lstm in self.lstms:
            out, _ = lstm(out, (h0, c0))

        # Apply the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

# Usage
num_layers = 1 # Adjust the depth of the model as needed
model = DeepLSTM(1, 4, num_layers)
model.to(device)

learning_rate = 0.001
num_epochs = 10
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

for epoch in range(num_epochs):
    train_one_epoch(epoch)
    accuracy()

with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

plt.plot(y_train, label='Actual Value')
plt.plot(predicted, label='Predicted Value')
plt.xlabel('Day')
plt.ylabel('Value')
plt.legend()
plt.show()

with torch.no_grad():
    predicted = model(X_test.to(device)).to('cpu').numpy()

plt.plot(y_test, label='Actual Value')
plt.plot(predicted, label='Predicted Value')
plt.xlabel('Day')
plt.ylabel('Value')
plt.legend()
plt.show()