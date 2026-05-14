
import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.loader import DataManager, TushareDataLoader
from backtest.engine import BacktestEngine, BacktestConfig
from strategy.limit_2_leader import Limit2LeaderStrategy

def run_strategy():
    # 1. 准备数据
    # 为了演示，我们使用最近一年的数据
    start_date = "20240101"
    end_date = "20241231"
    
    print(f"Loading data from {start_date} to {end_date}...")
    loader = TushareDataLoader()
    
    # 检查是否有数据，如果没有则下载（为了演示快速，这里假设用户已经有数据或者下载少量）
    # 实际运行时请确保数据已下载
    # loader.download_all(start_date, end_date) 
    
    dm = DataManager(loader)
    dm.load(start_date, end_date)
    
    if dm.price.empty:
        print("No data found. Please ensure data is downloaded.")
        return

    # 2. 初始化策略
    strategy = Limit2LeaderStrategy(n_stocks=5) # 测试用5只
    
    # 3. 运行策略生成权重
    print("Running strategy...")
    weights = strategy.run(dm)
    
    if weights.empty:
        print("No signals generated.")
        return

    # 4. 运行回测
    print("Running backtest...")
    config = BacktestConfig(
        initial_capital=1000000,
        commission_rate=0.0003,
        slippage=0.001
    )
    engine = BacktestEngine(config)
    result = engine.run(weights, dm)
    
    # 5. 输出结果
    print("\n" + "="*50)
    print("Backtest Results")
    print("="*50)
    print(f"Total Returns: {result.cumulative_returns.iloc[-1]:.2%}")
    print(f"Annualized Returns: {(1 + result.cumulative_returns.iloc[-1])**(252/len(result.cumulative_returns)) - 1:.2%}")
    print(f"Max Drawdown: {(result.net_value / result.net_value.cummax() - 1).min():.2%}")
    
    # 保存结果
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    result.portfolio_returns.to_csv(output_dir / "limit_2_leader_returns.csv")
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    run_strategy()
