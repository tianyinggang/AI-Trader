def update_data(symbols, lookback_days=5):
    """增量更新数据，避免重复下载全量数据"""
    # 1. 确定每个symbol最新的本地数据日期
    # 2. 只下载该日期之后的新数据
    # 3. 合并到已有数据中
    # 4. 触发特征重新计算