import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df, strategy="mean"):
    """
    处理缺失值。

    Args:
        df (pd.DataFrame): 数据集。
        strategy (str): 填充策略，可选 "mean", "median", "zero"。

    Returns:
        pd.DataFrame: 处理后的数据集。
    """
    if strategy == "mean":
        return df.fillna(df.mean())
    elif strategy == "median":
        return df.fillna(df.median())
    elif strategy == "zero":
        return df.fillna(0)
    else:
        raise ValueError("Unsupported strategy. Use 'mean', 'median', or 'zero'.")

def standardize_data(df, columns):
    """
    标准化指定列的数据。

    Args:
        df (pd.DataFrame): 数据集。
        columns (list): 需要标准化的列名列表。

    Returns:
        pd.DataFrame: 标准化后的数据集。
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df