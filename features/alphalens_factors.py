import pandas as pd
import numpy as np

def compute_momentum_factors(df, windows=[5, 20, 60]):
    """
    计算多周期动量因子。

    Args:
        df (pd.DataFrame): 包含价格数据的DataFrame，需包含 'Close' 列。
        windows (list): 动量计算的窗口期列表。

    Returns:
        pd.DataFrame: 包含动量因子的DataFrame。
    """
    momentum_factors = pd.DataFrame(index=df.index)
    for window in windows:
        momentum_factors[f"momentum_{window}"] = df["Close"].pct_change(periods=window)
    return momentum_factors

def compute_volatility_factors(df, windows=[20, 60]):
    """
    计算波动率相关因子。

    Args:
        df (pd.DataFrame): 包含价格数据的DataFrame，需包含 'close' 列。
        windows (list): 波动率计算的窗口期列表。

    Returns:
        pd.DataFrame: 包含波动率因子的DataFrame。
    """
    volatility_factors = pd.DataFrame(index=df.index)
    for window in windows:
        volatility_factors[f"volatility_{window}"] = df["Close"].rolling(window=window).std()
    return volatility_factors

def compute_fundamental_factors(df):
    """
    计算基本面因子。

    Args:
        df (pd.DataFrame): 包含财务数据的DataFrame，需包含 'pe', 'pb', 'roe' 等列。

    Returns:
        pd.DataFrame: 包含基本面因子的DataFrame。
    """
    fundamental_factors = pd.DataFrame(index=df.index)
    if "pe" in df.columns:
        fundamental_factors["inverse_pe"] = 1 / df["pe"].replace(0, np.nan)  # 避免除以零
    if "pb" in df.columns:
        fundamental_factors["inverse_pb"] = 1 / df["pb"].replace(0, np.nan)
    if "roe" in df.columns:
        fundamental_factors["roe"] = df["roe"]
    return fundamental_factors