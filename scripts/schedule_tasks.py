from apscheduler.schedulers.blocking import BlockingScheduler

def setup_scheduler():
    """设置定时任务"""
    scheduler = BlockingScheduler(timezone="US/Eastern")
    
    # 每个交易日14:30(美东时间)运行每日预测
    scheduler.add_job(
        'scripts.daily_prediction:main',
        'cron', 
        day_of_week='mon-fri',
        hour=14, 
        minute=30
    )
    
    # 每周日晚运行周度再平衡
    scheduler.add_job(
        'trading.rebalancer:weekly_rebalance',
        'cron',
        day_of_week='sun',
        hour=20
    )
    
    # 每月最后一个周日运行月度再平衡
    scheduler.add_job(
        'trading.rebalancer:monthly_rebalance',
        'cron',
        day='last sun'
    )
    
    return scheduler