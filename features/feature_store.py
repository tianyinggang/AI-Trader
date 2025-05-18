from features.utils import handle_missing_values, standardize_data
from features.tech_indicators import calculate_sma, calculate_rsi, calculate_macd
from features.alphalens_factors import compute_momentum_factors, compute_volatility_factors, compute_fundamental_factors
import pandas as pd
def process_features(raw_features):
    """
    特征处理流水线。

    Args:
        raw_features (pd.DataFrame): 原始特征数据。

    Returns:
        pd.DataFrame: 处理后的特征集。
    """
    # 1. 处理缺失值
    processed_features = handle_missing_values(raw_features, strategy="mean")

    # 2. 添加技术指标
    processed_features["SMA_14"] = calculate_sma(processed_features, column="Close", window=14)
    processed_features["RSI_14"] = calculate_rsi(processed_features, column="Close", window=14)
    macd = calculate_macd(processed_features, column="Close")
    processed_features = pd.concat([processed_features, macd], axis=1)

    # 3. 添加动量因子
    momentum_factors = compute_momentum_factors(processed_features, windows=[5, 20, 60])
    processed_features = pd.concat([processed_features, momentum_factors], axis=1)

    # 4. 添加波动率因子
    volatility_factors = compute_volatility_factors(processed_features, windows=[20, 60])
    processed_features = pd.concat([processed_features, volatility_factors], axis=1)

    # 5. 添加基本面因子
    fundamental_factors = compute_fundamental_factors(processed_features)
    processed_features = pd.concat([processed_features, fundamental_factors], axis=1)

    # 6. 标准化
    columns_to_standardize = [
        "SMA_14", "RSI_14", "MACD", "Signal",
        *momentum_factors.columns, *volatility_factors.columns, *fundamental_factors.columns
    ]
    processed_features = standardize_data(processed_features, columns=columns_to_standardize)

    # 7. 返回处理后的特征集
    return processed_features