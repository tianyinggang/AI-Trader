import yfinance as yf
import os
def download_nasdaq100_history(start_date=None, end_date=None):
    """下载纳斯达克100指数及成分股的全量历史数据"""
    # 确保目标目录存在
    output_dir = "./data/raw"
    os.makedirs(output_dir, exist_ok=True)

    # 使用yfinance下载QQQ和成分股
    # 使用Alpha Vantage作为备选数据源
    # 返回DataFrame或存储到data/raw/
    qqq = yf.Ticker("QQQ")
    qqq_history = qqq.history(period="max")
    qqq_history.to_csv("./data/raw/qqq_history.csv")

def download_crypto_history(symbol="BTC-USD", start_date=None, end_date=None):
    """下载加密货币历史数据（为后期扩展做准备）"""
    # 实现类似上面的功能，但针对加密货币
#text download_nasdaq100_history()
# download_crypto_history()
if __name__ == "__main__":
    download_nasdaq100_history()