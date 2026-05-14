"""
示例脚本 - 完整回测流程演示

本脚本演示量化回测系统的完整流程：
1. 创建模拟数据
2. 计算因子
3. 运行多个策略
4. 执行回测
5. 生成绩效报告
6. 策略对比分析

运行方式：
    cd quant_backtest
    python examples/demo_backtest.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from factor import (
    CircMarketCapFactor,
    MarketCapRankFactor,
    MomentumFactor,
    FactorUtils,
)

from strategy import (
    SmallCapStrategy,
    SmallCapValueStrategy,
    SmallCapMomentumStrategy,
    SmallCapReversalStrategy,
    create_small_cap_strategy,
)

from backtest import (
    BacktestEngine,
    BacktestConfig,
    BacktestReport,
    SimpleBacktest,
    calc_all_metrics,
    compare_strategies,
    print_strategy_comparison,
)


# ==================== 创建模拟数据 ====================

def create_mock_data(n_days: int = 500, n_stocks: int = 200):
    """
    创建模拟数据用于测试
    
    Args:
        n_days: 交易日数量（约2年）
        n_stocks: 股票数量
    
    Returns:
        MockDataManager实例
    """
    np.random.seed(42)
    
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')
    stocks = [f'{i:06d}.SZ' if i <= n_stocks//2 else f'{i:06d}.SH' 
              for i in range(1, n_stocks + 1)]
    
    # 模拟价格数据
    # 添加一些趋势和波动
    base_returns = np.random.randn(n_days, n_stocks) * 0.025
    
    # 给小市值股票添加一些超额收益（模拟小市值效应）
    size_premium = np.linspace(0.0002, -0.0001, n_stocks)  # 小市值股票有正超额
    base_returns += size_premium
    
    # 添加市场因子
    market_factor = np.random.randn(n_days) * 0.015
    base_returns += market_factor.reshape(-1, 1) * 0.5
    
    prices = 50 * np.exp(np.cumsum(base_returns, axis=0))
    
    # 模拟市值数据
    base_mcap = np.random.lognormal(mean=12, sigma=1.5, size=n_stocks)
    mcap_changes = np.random.randn(n_days, n_stocks) * 0.008
    mcap = base_mcap * np.exp(np.cumsum(mcap_changes, axis=0))
    
    # 流通市值
    circ_ratio = np.random.uniform(0.6, 0.9, size=n_stocks)
    circ_mv = mcap * circ_ratio
    
    # 估值数据
    pe = np.random.uniform(5, 80, (n_days, n_stocks))
    pb = np.random.uniform(0.5, 8, (n_days, n_stocks))
    ps = np.random.uniform(0.5, 15, (n_days, n_stocks))
    
    # 成交量（用于判断停牌）
    volume = np.random.uniform(1000, 100000, (n_days, n_stocks))
    # 随机设置一些停牌（约1%）
    suspend_mask = np.random.random((n_days, n_stocks)) < 0.01
    volume[suspend_mask] = 0
    
    # 创建DataFrame
    mock_price = pd.DataFrame(prices, index=dates, columns=stocks)
    mock_mcap = pd.DataFrame(mcap, index=dates, columns=stocks)
    mock_circ_mv = pd.DataFrame(circ_mv, index=dates, columns=stocks)
    mock_pe = pd.DataFrame(pe, index=dates, columns=stocks)
    mock_pb = pd.DataFrame(pb, index=dates, columns=stocks)
    mock_ps = pd.DataFrame(ps, index=dates, columns=stocks)
    mock_volume = pd.DataFrame(volume, index=dates, columns=stocks)
    
    # 创建基准收益（模拟沪深300）
    benchmark_returns = pd.Series(
        np.random.randn(n_days) * 0.012 + 0.0001,
        index=dates
    )
    
    class MockDataManager:
        @property
        def price(self):
            return mock_price
        
        @property
        def open(self):
            return mock_price * np.random.uniform(0.99, 1.01, mock_price.shape)
        
        @property
        def market_cap(self):
            return mock_mcap
        
        @property
        def circ_mv(self):
            return mock_circ_mv
        
        def get_matrix(self, field, source):
            data = {
                'pe_ttm': mock_pe,
                'pe': mock_pe,
                'pb': mock_pb,
                'ps_ttm': mock_ps,
                'ps': mock_ps,
                'total_mv': mock_mcap,
                'circ_mv': mock_circ_mv,
                'vol': mock_volume,
            }
            return data.get(field, pd.DataFrame())
    
    return MockDataManager(), benchmark_returns


# ==================== 主程序 ====================

def main():
    print("=" * 70)
    print("量化回测系统 - 完整流程演示")
    print("=" * 70)
    
    # 1. 创建模拟数据
    print("\n【1. 创建模拟数据】")
    dm, benchmark_returns = create_mock_data(n_days=500, n_stocks=200)
    print(f"数据范围: {dm.price.index[0].date()} ~ {dm.price.index[-1].date()}")
    print(f"交易天数: {len(dm.price)}")
    print(f"股票数量: {len(dm.price.columns)}")
    
    # 2. 运行多个策略
    print("\n【2. 运行策略】")
    strategies = {}
    
    # 策略1: 纯小市值
    print("\n  运行策略: 纯小市值...")
    strategy1 = SmallCapStrategy(
        n_stocks=30,
        rebalance_freq="weekly",
        max_market_cap=None,
        name="pure_small_cap"
    )
    weights1 = strategy1.run(dm)
    strategies['纯小市值'] = (strategy1, weights1)
    
    # 策略2: 小市值+价值
    print("  运行策略: 小市值+价值...")
    strategy2 = SmallCapValueStrategy(
        n_stocks=30,
        size_weight=0.6,
        value_weight=0.4,
        rebalance_freq="weekly",
        max_market_cap=None,
        name="small_cap_value"
    )
    weights2 = strategy2.run(dm)
    strategies['小市值+价值'] = (strategy2, weights2)
    
    # 策略3: 小市值+动量过滤
    print("  运行策略: 小市值+动量过滤...")
    strategy3 = SmallCapMomentumStrategy(
        n_stocks=30,
        momentum_window=20,
        momentum_threshold=-0.05,
        rebalance_freq="weekly",
        max_market_cap=None,
        name="small_cap_momentum"
    )
    weights3 = strategy3.run(dm)
    strategies['小市值+动量'] = (strategy3, weights3)
    
    # 策略4: 小市值+反转
    print("  运行策略: 小市值+反转...")
    strategy4 = SmallCapReversalStrategy(
        n_stocks=30,
        reversal_window=5,
        size_weight=0.5,
        reversal_weight=0.5,
        rebalance_freq="weekly",
        max_market_cap=None,
        name="small_cap_reversal"
    )
    weights4 = strategy4.run(dm)
    strategies['小市值+反转'] = (strategy4, weights4)
    
    # 3. 执行回测
    print("\n【3. 执行回测】")
    
    # 配置回测参数
    config = BacktestConfig(
        commission_rate=0.0003,
        slippage=0.001,
        stamp_duty=0.001,
        trade_delay=1,
        apply_limit_rules=True,
        apply_suspend_rules=True,
        initial_capital=1000000,
    )
    
    engine = BacktestEngine(config)
    backtest_results = {}
    
    for name, (strategy, weights) in strategies.items():
        print(f"\n  回测策略: {name}")
        result = engine.run(weights, dm, benchmark_returns)
        backtest_results[name] = result
    
    # 4. 生成报告
    print("\n【4. 绩效报告】")
    
    for name, result in backtest_results.items():
        report = BacktestReport(result, name)
        report.print_summary()
    
    # 5. 策略对比
    print("\n【5. 策略对比】")
    print_strategy_comparison(backtest_results)
    
    # 6. 详细分析
    print("\n【6. 详细分析】")
    
    # 最佳策略
    best_strategy = max(backtest_results.items(), 
                        key=lambda x: calc_all_metrics(x[1].portfolio_returns).sharpe_ratio)
    print(f"\n最佳夏普比率策略: {best_strategy[0]}")
    
    best_report = BacktestReport(best_strategy[1], best_strategy[0])
    
    # 年度收益
    print("\n年度收益率:")
    yearly = best_report.get_yearly_returns()
    for year, ret in yearly.items():
        print(f"  {year}: {ret:>8.2%}")
    
    # 回撤分析
    print("\n回撤分析:")
    dd_analysis = best_report.get_drawdown_analysis()
    for k, v in dd_analysis.items():
        print(f"  {k}: {v}")
    
    # 月度收益表
    print("\n月度收益率表格:")
    monthly = best_report.get_monthly_returns()
    # 格式化为百分比
    monthly_pct = monthly.applymap(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    print(monthly_pct.to_string())
    
    # 7. 简化回测对比
    print("\n【7. 简化回测验证】")
    
    # 使用SimpleBacktest快速验证
    returns = dm.price.pct_change()
    
    for name, (strategy, weights) in strategies.items():
        nav = SimpleBacktest.run(weights, returns, cost_rate=0.003, delay=1)
        total_ret = nav.iloc[-1] - 1
        print(f"  {name}: {total_ret:.2%}")
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
