# 🏷️ nasdaq‑ai‑trader

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()

> **AI‑Driven Nasdaq100 & BTC Trader**  
> 一个端到端的量化交易系统原型，涵盖：  
> - **数据模块**：yfinance / AlphaVantage 原始与增量下载、Parquet/HDF5 存储  
> - **因子工程**：动量、波动率、基本面、技术指标、AlphaLens 分析  
> - **模型**：强化学习 (PPO, DDPG, SAC, TD3) + 传统 ML (LightGBM, XGBoost, Transformer)  
> - **回测引擎**：交易逻辑、成本、滑点、绩效指标（年化、Sharpe、回撤）  
> - **实盘信号**：每日美东 14:30 生成 CSV/HTML 报告，并通过 Email/Slack 推送  
> - **调度**：APScheduler 支持日/周/月任务  

---

## 📸 系统架构概览

![系统架构示意图](https://raw.githubusercontent.com/yourname/nasdaq-ai-trader/main/docs/architecture.png)

<details>
<summary>点击查看目录结构</summary>

```
nasdaq-ai-trader/
├── README.md                        
├── requirements.txt                 
├── config/                          
│   ├── default.yaml                 
│   └── env.yaml                     
├── data/                            
│   ├── raw/                         
│   ├── processed/                   
│   ├── download.py                  
│   ├── update.py                    
│   └── utils.py                     
├── features/                        
│   ├── alphalens_factors.py         
│   ├── tech_indicators.py           
│   ├── feature_store.py             
│   └── utils.py                     
├── models/                          
│   ├── rl/                          
│   ├── ml/                          
│   └── utils.py                     
├── backtest/                        
│   ├── engine.py                    
│   ├── strategy.py                  
│   ├── metrics.py                   
│   └── report.py                    
├── trading/                         
│   ├── signal_generator.py          
│   ├── rebalancer.py                
│   └── notification.py              
├── scripts/                         
│   ├── train_rl.py                  
│   ├── train_ml.py                  
│   ├── run_backtest.py              
│   ├── daily_prediction.py          
│   └── schedule_tasks.py            
├── notebooks/                       
│   ├── 01_Data_Exploration.ipynb    
│   ├── 02_Feature_Engineering.ipynb 
│   ├── 03_RL_Models.ipynb           
│   ├── 04_ML_Models.ipynb           
│   ├── 05_Backtest_Analysis.ipynb   
│   └── 06_Signal_Analysis.ipynb     
├── outputs/                         
│   ├── models/                      
│   ├── backtest/                    
│   └── signals/                     
└── utils/                           
    ├── logger.py                    
    ├── scheduler.py                 
    └── settings.py                  
```
</details>

---

## 🚀 快速开始

1. **克隆仓库**  
   ```bash
   git clone https://github.com/yourname/nasdaq-ai-trader.git
   cd nasdaq-ai-trader
   ```

2. **创建虚拟环境 & 安装依赖**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **配置 API Keys**  
   编辑 `config/env.yaml`：
   ```yaml
   ALPHA_VANTAGE_API_KEY: "your_alpha_vantage_key"
   EMAIL_SMTP_SERVER: "smtp.example.com"
   EMAIL_USERNAME: "you@example.com"
   EMAIL_PASSWORD: "password"
   SLACK_WEBHOOK_URL: "https://hooks.slack.com/..."
   ```

4. **下载历史数据**  
   ```bash
   python data/download.py --start 2015-01-01 --end 2025-01-01
   ```

5. **运行回测**  
   ```bash
   python scripts/run_backtest.py --start 2020-01-01 --end 2024-12-31
   ```
   ![Backtest Result](https://raw.githubusercontent.com/yourname/nasdaq-ai-trader/main/docs/backtest_example.png)

6. **生成每日信号**  
   ```bash
   python scripts/daily_prediction.py
   ```

---

## 📚 核心模块说明

| 模块          | 功能                                                         |
|---------------|--------------------------------------------------------------|
| **data/**     | 数据下载、增量更新、Parquet/HDF5 存储                         |
| **features/** | 动量/波动率/基本面因子、技术指标、AlphaLens 分析             |
| **models/**   | RL (PPO/DDPG/SAC/TD3)、ML (LightGBM/XGBoost/Transformer)     |
| **backtest/** | 回测引擎、策略、绩效指标、报告                                 |
| **trading/**  | 信号生成、再平衡、Email/Slack 推送                           |
| **scripts/**  | 一键训练/回测/信号、任务调度                                  |

---

## 🔧 配置项（`config/default.yaml`）

```yaml
data:
  symbols:
    - QQQ
    - AAPL
    - MSFT
  crypto:
    - BTC-USD

backtest:
  initial_capital: 100000
  commission: 0.001
  slippage: 0.001

models:
  rl:
    ppo:
      lr: 3e-4
      gamma: 0.99
  ml:
    lgbm:
      num_leaves: 31
      learning_rate: 0.05
```

---

## 📈 示例结果

- **因子分析可视化**  
  ![AlphaLens 因子分析](https://raw.githubusercontent.com/yourname/nasdaq-ai-trader/main/docs/alphalens_factors.png)

- **信号热力图**  
  ![Signal Heatmap](https://raw.githubusercontent.com/yourname/nasdaq-ai-trader/main/docs/signal_heatmap.png)

---
## TODO
- [ ] 划分 Train/Val/Test (如 70%/15%/15%): 
- [ ] Task 2 ：增加特征工程的文档，解释什么意思 --DDL: 19/05
- [ ] Task 3：100%的准确率就是因为训练次数太多，但是数据又太少，导致的。
- [ ] Task 4: 利用yfinance将能下的数据都下载下来。 DDL:20/05
应该多加入其他特征。
建议：

* 交叉验证：使用时间序列交叉验证，而非简单的随机划分
* 特征工程审查：确保特征不包含未来信息
* 模型复杂度调整：尝试降低模型复杂度，加强正则化
* 扩大测试集：使用更大、更多样化的测试数据评估模型
* 回测验证：在真实或模拟的交易环境中回测模型

总结：虽然指标看起来非常好，但100%的方向准确率值得怀疑。在实际应用前，建议进行更严格的验证和测试。

## 📜 许可证

本项目基于 MIT 许可证开源，详情见 [LICENSE](LICENSE)。

---

## 🤝 致谢

- [yfinance](https://github.com/ranaroussi/yfinance)  
- [Alpha Vantage](https://www.alphavantage.co/)  
- [PyTorch](https://pytorch.org/)  
- [AlphaLens](https://github.com/quantopian/alphalens)  
- [APScheduler](https://apscheduler.readthedocs.io/)  

---

> 如果你有任何问题或建议，欢迎提交 Issue 或 Pull Request！  
> Happy Trading! 🚀