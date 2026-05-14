"""
量化回测系统

一个完整的A股量化回测框架，包含：
- 数据模块 (data): 从Tushare获取数据
- 因子模块 (factor): 计算各类量化因子
- 策略模块 (strategy): 实现各种量化策略
- 回测模块 (backtest): 策略回测和评估（待开发）

使用示例:
    from quant_backtest.data import TushareDataLoader, DataManager
    from quant_backtest.strategy import SmallCapStrategy
    
    # 加载数据
    loader = TushareDataLoader()
    dm = DataManager(loader)
    dm.load("20200101", "20231231")
    
    # 运行策略
    strategy = SmallCapStrategy(n_stocks=30)
    weights = strategy.run(dm)
"""

__version__ = "0.3.0"
__author__ = "Quant Team"
