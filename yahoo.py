# yfinance requires pandas 1.3.5, breaks with 1.4.0
import yfinance as yf
import pandas as pd
import os

today = pd.Timestamp.today()
dates = pd.bdate_range(today - pd.Timedelta(days=730), today)

# Fetch 30 days of minute data for all valid stock symbols (symbols as of 2023-07-14).
# with open("valid_stocks.txt", "r") as f:
    # tickers = f.read().splitlines()

tickers = ['VNQ']
full_data = yf.download(tickers[0], group_by='Ticker', start=dates[0].date(), end=dates[-1].date())


# Select the first row
one_row_df = full_data.T.iloc[4:5]
one_row_df.to_csv(f'VNQ.csv', float_format='%.2f')



# Alternatively, you can directly print the transposed DataFrame


# for dt_date in dates:
#     sdate = dt_date.date()
#     edate = (dt_date + pd.Timedelta(days=1)).date()
#     for ticker in tickers:
#         os.makedirs(f'intraday_data/stocks/{sdate}', exist_ok=True)
#         try:
#             data = yf.download(ticker, group_by="Ticker", start=sdate, end=edate, interval="1d")
#             full_data = full_data.add(data)
#             #data.to_csv(f'intraday_data/stocks/{sdate}/{ticker}.csv', float_format='%.2f')
#         except Exception as e:
#             print(f"Caught {e}")






