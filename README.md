**GitHub Description (optional)**  
> **AI‑Driven Nasdaq100 & Bitcoin Trading System**  
> A full‑stack Python framework for end‑to‑end data ingestion, factor engineering, ML/RL modeling, backtesting, live signal generation and notification—built for research, demo and production.  

---

# 🏷️ nasdaq‑ai‑trader

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()

> **AI‑Driven Nasdaq100 & BTC Trader**  
> An end‑to‑end quantitative trading prototype covering:  
> - **Data**: yfinance & Alpha Vantage raw & incremental download, Parquet/HDF5 storage  
> - **Factors**: momentum, volatility, fundamental, technical indicators & AlphaLens analysis  
> - **Models**: RL (PPO, DDPG, SAC, TD3) + ML (LightGBM, XGBoost, Transformer)  
> - **Backtesting**: trade logic, transaction costs, slippage, performance metrics (annualized return, Sharpe, drawdown)  
> - **Live Signals**: daily US‑market signals at 14:30 ET, CSV/HTML reports + Email/Slack delivery  
> - **Scheduling**: APScheduler for daily/weekly/monthly jobs  

---

## 📸 System Architecture

![System Architecture](https://raw.githubusercontent.com/yourname/nasdaq-ai-trader/main/docs/architecture.png)

<details>
<summary>Click to expand folder structure</summary>

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

## 🚀 Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourname/nasdaq-ai-trader.git
   cd nasdaq-ai-trader
   ```

2. **Set up virtual environment & install dependencies**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure API keys**  
   Edit `config/env.yaml`:
   ```yaml
   ALPHA_VANTAGE_API_KEY: "your_alpha_vantage_key"
   EMAIL_SMTP_SERVER: "smtp.example.com"
   EMAIL_USERNAME: "you@example.com"
   EMAIL_PASSWORD: "password"
   SLACK_WEBHOOK_URL: "https://hooks.slack.com/..."
   ```

4. **Download historical data**  
   ```bash
   python data/download.py --start 2015-01-01 --end 2025-01-01
   ```

5. **Run backtest**  
   ```bash
   python scripts/run_backtest.py --start 2020-01-01 --end 2024-12-31
   ```
   ![Backtest Example](https://raw.githubusercontent.com/yourname/nasdaq-ai-trader/main/docs/backtest_example.png)

6. **Generate daily signals**  
   ```bash
   python scripts/daily_prediction.py
   ```

---

## 📚 Core Modules

| Module         | Purpose                                                       |
| -------------- | ------------------------------------------------------------- |
| **data/**      | Data ingestion, incremental updates, Parquet/HDF5 storage     |
| **features/**  | Momentum/volatility/fundamental factors, technical indicators |
| **models/**    | RL (PPO/DDPG/SAC/TD3), ML (LightGBM/XGBoost/Transformer)      |
| **backtest/**  | Backtesting engine, strategies, performance metrics, reports  |
| **trading/**   | Signal generation, rebalancing, Email/Slack notification      |
| **scripts/**   | One‑click training/backtest/signal & job scheduling           |

---

## 🔧 Configuration (`config/default.yaml`)

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

## 📈 Sample Outputs

- **AlphaLens Factor Analysis**  
  ![AlphaLens Analysis](https://raw.githubusercontent.com/yourname/nasdaq-ai-trader/main/docs/alphalens_factors.png)

- **Signal Heatmap**  
  ![Signal Heatmap](https://raw.githubusercontent.com/yourname/nasdaq-ai-trader/main/docs/signal_heatmap.png)

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance)  
- [Alpha Vantage](https://www.alphavantage.co/)  
- [PyTorch](https://pytorch.org/)  
- [AlphaLens](https://github.com/quantopian/alphalens)  
- [APScheduler](https://apscheduler.readthedocs.io/)  

---

> Questions, feedback or contributions are welcome—feel free to open an issue or pull request!  
> Happy trading! 🚀



ensor([[[0.0504],
         [0.0508],
         [0.0487],
         ...,
         [0.0479],
         [0.0497],
         [0.0538]],

        [[0.0508],
         [0.0487],
         [0.0511],
         ...,
         [0.0497],
         [0.0538],
         [0.0561]],

        [[0.0487],
         [0.0511],
         [0.0518],
         ...,
         [0.0538],
         [0.0561],
         [0.0573]],

        ...,
...
    lr: 0.001
    maximize: False
    weight_decay: 0
) MSELoss()