import pandas as pd

def calculate_sma(df, column="Close", window=14):
    """
    计算简单移动平均线 (SMA)。

    Args:
        df (pd.DataFrame): 数据集。
        column (str): 计算SMA的列名。
        window (int): 移动窗口大小。

    Returns:
        pd.Series: SMA值。
    """
    return df[column].rolling(window=window).mean()

def calculate_rsi(df, column="Close", window=14):
    """
    计算相对强弱指数 (RSI)。

    Args:
        df (pd.DataFrame): 数据集。
        column (str): 计算RSI的列名。
        window (int): 移动窗口大小。

    Returns:
        pd.Series: RSI值。
    """
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, column="Close", short_window=12, long_window=26, signal_window=9):
    """
    计算移动平均线收敛/发散指标 (MACD)。

    Args:
        df (pd.DataFrame): 数据集。
        column (str): 计算MACD的列名。
        short_window (int): 短期EMA窗口。
        long_window (int): 长期EMA窗口。
        signal_window (int): 信号线窗口。

    Returns:
        pd.DataFrame: 包含MACD线和信号线的DataFrame。
    """
    short_ema = df[column].ewm(span=short_window, adjust=False).mean()
    long_ema = df[column].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return pd.DataFrame({"MACD": macd, "Signal": signal})