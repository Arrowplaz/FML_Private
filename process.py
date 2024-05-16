import numpy as np
import pandas as pd
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler


#******CHANGE THIS*******
# file_path = "/Users/anagireddygari/Desktop/Econ Final/FML_Private/test.csv"
file_path = 'XLRE.csv'
# file_path = "/Users/anagireddygari/Desktop/Econ Final/FML_Private/VNQ.csv"


#*******THIS IS NECESSARY FOR READING THE STOCK DATA
df = pd.read_csv(file_path)


dates = []
#Iterates over each row to grab the date
for row,index in df.iterrows():
    for i,date in enumerate(row): 
        if i >= 5: #Skips the first couple columns
            dates.append(date)
    break #Break lets us stop after 1 row



iterator = df.iterrows()
next(iterator)# Skip the first row
house_prices = [] # list of lists - each list is sequence of house prices
train_prices = []
train_prices_labels = []
test_prices = []
test_prices_labels = []
j = 1
for index, row in iterator: #Iterates over each row
    total_time_series = [] #Holder for prices in each row
    for i, price in enumerate(index):
        if i >= 5:# Skip the first 5 iterations
            price = float(price)
            total_time_series.append(price)
    #Split the training data
    if j<=500: # training data
        train_prices.append(total_time_series[:150])
        train_prices_labels.append(total_time_series[150:])
    else: # test data
        test_prices.append(total_time_series[:150])
        test_prices_labels.append(total_time_series[150:])
    house_prices.append(total_time_series)
    j += 1


# create additional columns for n_steps previous time steps
# now each row represents historical house values
def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Price(t-{i})'] = df['Price'].shift(i)

    df.dropna(inplace=True)

    return df

# normalise the prices and create dataframe with historical prices for 7 day lookback
def prepare_data(row, lookback):

    data = {
        'Date': dates,
        'Price': house_prices[row]
    }

    df = pd.DataFrame(data)
    shifted_df = prepare_dataframe_for_lstm(df, lookback)

    shifted_df_as_np = shifted_df.to_numpy()


    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]

    return X, y
