import yfinance as yf
import os
import pandas as pd
from datetime import datetime

def download_data(symbol, start_date, end_date):
    """
    下载指定symbol的历史数据。

    Args:
        symbol (str): 股票或资产的符号。
        start_date (datetime): 数据开始日期。
        end_date (datetime): 数据结束日期。

    Returns:
        pd.DataFrame: 包含下载数据的DataFrame，包含日期和价格等信息。
    """
    print(f"Downloading data for {symbol} from {start_date} to {end_date}...")
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if history.empty:
            print(f"No data found for {symbol} in the given date range.")
            return pd.DataFrame()
        history.reset_index(inplace=True)
        history.rename(columns={"Date": "date"}, inplace=True)
        return history
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return pd.DataFrame()

def download_nasdaq100_history():
    """
    下载纳斯达克100指数及其成分股的全量历史数据。
    """
    output_dir = "./data/raw"
    os.makedirs(output_dir, exist_ok=True)

    # 下载QQQ指数历史数据
    qqq = yf.Ticker("QQQ")
    qqq_history = qqq.history(period="max")
    qqq_history.to_csv(os.path.join(output_dir, "qqq_history.csv"))
    print("QQQ history downloaded successfully.")

def download_crypto_history(symbol="BTC-USD", start_date=None, end_date=None):
    """
    下载加密货币历史数据。

    Args:
        symbol (str): 加密货币符号，默认 "BTC-USD"。
        start_date (datetime): 数据开始日期。
        end_date (datetime): 数据结束日期。
    """
    output_dir = "./data/raw/crypto"
    os.makedirs(output_dir, exist_ok=True)

    if start_date is None:
        start_date = datetime(2010, 1, 1)  # 默认从2010年开始
    if end_date is None:
        end_date = datetime.now()

    crypto_data = download_data(symbol, start_date, end_date)
    if not crypto_data.empty:
        crypto_data.to_csv(os.path.join(output_dir, f"{symbol}_history.csv"), index=False)
        print(f"{symbol} history downloaded successfully.")
    else:
        print(f"No data downloaded for {symbol}.")

if __name__ == "__main__":
    # 示例：下载纳斯达克100指数历史数据
    download_nasdaq100_history()

    # 示例：下载比特币历史数据
    download_crypto_history("BTC-USD")