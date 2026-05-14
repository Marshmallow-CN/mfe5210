"""
示例脚本 - 小市值策略演示

本脚本演示如何使用量化回测系统：
1. 使用模拟数据测试（不需要Tushare API）
2. 计算市值因子
3. 运行小市值策略
4. 分析策略结果

运行方式：
    cd quant_backtest
    python examples/demo_small_cap.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from factor import (
    MarketCapFactor,
    CircMarketCapFactor,
    MarketCapRankFactor,
    FactorUtils,
    PEFactor,
    MomentumFactor,
)

from strategy import (
    SmallCapStrategy,
    SmallCapValueStrategy,
    SmallCapMomentumStrategy,
    create_small_cap_strategy,
    StrategyConfig,
)


# ==================== 创建模拟数据 ====================

def create_mock_data(n_days: int = 250, n_stocks: int = 100):
    """
    创建模拟数据用于测试
    
    Args:
        n_days: 交易日数量
        n_stocks: 股票数量
    
    Returns:
        MockDataManager实例
    """
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')  # 工作日
    stocks = [f'{i:06d}.SZ' if i <= n_stocks//2 else f'{i:06d}.SH' 
              for i in range(1, n_stocks + 1)]
    
    # 模拟价格数据（对数正态分布）
    returns = np.random.randn(n_days, n_stocks) * 0.02  # 日收益率约2%标准差
    prices = 50 * np.exp(np.cumsum(returns, axis=0))  # 起始价格50
    
    # 模拟市值数据（分布不均，有小盘有大盘）
    base_mcap = np.random.lognormal(mean=12, sigma=1.5, size=n_stocks)  # 万元
    mcap_changes = np.random.randn(n_days, n_stocks) * 0.01
    mcap = base_mcap * np.exp(np.cumsum(mcap_changes, axis=0))
    
    # 流通市值约为总市值的60%-90%
    circ_ratio = np.random.uniform(0.6, 0.9, size=n_stocks)
    circ_mv = mcap * circ_ratio
    
    # 模拟估值数据
    pe = np.random.uniform(5, 100, (n_days, n_stocks))
    pe[pe < 0] = np.nan  # 部分亏损股票
    
    pb = np.random.uniform(0.5, 10, (n_days, n_stocks))
    ps = np.random.uniform(0.5, 20, (n_days, n_stocks))
    
    # 创建DataFrame
    mock_price = pd.DataFrame(prices, index=dates, columns=stocks)
    mock_mcap = pd.DataFrame(mcap, index=dates, columns=stocks)
    mock_circ_mv = pd.DataFrame(circ_mv, index=dates, columns=stocks)
    mock_pe = pd.DataFrame(pe, index=dates, columns=stocks)
    mock_pb = pd.DataFrame(pb, index=dates, columns=stocks)
    mock_ps = pd.DataFrame(ps, index=dates, columns=stocks)
    
    # 创建MockDataManager
    class MockDataManager:
        @property
        def price(self):
            return mock_price
        
        @property
        def open(self):
            return mock_price * np.random.uniform(0.98, 1.02, mock_price.shape)
        
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
            }
            return data.get(field, pd.DataFrame())
    
    return MockDataManager()


# ==================== 因子测试 ====================

def test_factors(dm):
    """测试因子计算"""
    print("\n" + "="*60)
    print("因子测试")
    print("="*60)
    
    # 1. 市值因子
    print("\n【1. 市值因子】")
    mcap_factor = CircMarketCapFactor(negative=True)
    mcap_values = mcap_factor.compute(dm)
    print(f"因子形状: {mcap_values.shape}")
    print(f"最新截面前5只股票因子值:")
    print(mcap_values.iloc[-1].nlargest(5))
    
    # 2. 市值排名因子
    print("\n【2. 市值排名因子】")
    rank_factor = MarketCapRankFactor(negative=True)
    rank_values = rank_factor.compute(dm)
    print(f"最新截面排名前5（最小市值）:")
    print(rank_values.iloc[-1].nlargest(5))
    
    # 3. 动量因子
    print("\n【3. 动量因子】")
    mom_factor = MomentumFactor(window=20)
    mom_values = mom_factor.compute(dm)
    print(f"最新截面动量前5:")
    print(mom_values.iloc[-1].nlargest(5))
    
    return mcap_values, rank_values, mom_values


# ==================== 策略测试 ====================

def test_strategies(dm):
    """测试策略运行"""
    print("\n" + "="*60)
    print("策略测试")
    print("="*60)
    
    results = {}
    
    # 1. 纯小市值策略
    print("\n【1. 纯小市值策略】")
    strategy1 = SmallCapStrategy(
        n_stocks=20,
        rebalance_freq="weekly",
        max_market_cap=None,  # 不做市值限制，因为是模拟数据
    )
    weights1 = strategy1.run(dm)
    
    print(f"权重矩阵形状: {weights1.shape}")
    print(f"平均持股数: {(weights1 > 0).sum(axis=1).mean():.1f}")
    print(f"平均换手率: {strategy1.get_turnover().mean():.2%}")
    
    results['pure_small_cap'] = strategy1
    
    # 2. 小市值+价值策略
    print("\n【2. 小市值+价值策略】")
    strategy2 = SmallCapValueStrategy(
        n_stocks=20,
        size_weight=0.5,
        value_weight=0.5,
        rebalance_freq="weekly",
        max_market_cap=None,
    )
    weights2 = strategy2.run(dm)
    
    print(f"权重矩阵形状: {weights2.shape}")
    print(f"平均持股数: {(weights2 > 0).sum(axis=1).mean():.1f}")
    print(f"平均换手率: {strategy2.get_turnover().mean():.2%}")
    
    results['small_cap_value'] = strategy2
    
    # 3. 小市值+动量过滤策略
    print("\n【3. 小市值+动量过滤策略】")
    strategy3 = SmallCapMomentumStrategy(
        n_stocks=20,
        momentum_window=20,
        momentum_threshold=-0.1,  # 过滤掉跌幅超过10%的
        rebalance_freq="weekly",
        max_market_cap=None,
    )
    weights3 = strategy3.run(dm)
    
    print(f"权重矩阵形状: {weights3.shape}")
    print(f"平均持股数: {(weights3 > 0).sum(axis=1).mean():.1f}")
    print(f"平均换手率: {strategy3.get_turnover().mean():.2%}")
    
    results['small_cap_momentum'] = strategy3
    
    return results


# ==================== 策略比较 ====================

def compare_strategies(strategies: dict, dm):
    """比较不同策略的表现"""
    print("\n" + "="*60)
    print("策略比较")
    print("="*60)
    
    # 获取价格数据计算收益
    price = dm.price
    returns = price.pct_change().fillna(0)
    
    strategy_returns = {}
    
    for name, strategy in strategies.items():
        weights = strategy.weights
        
        # 计算策略收益：当日权重 * 当日收益
        # 注意：权重是截面时点的，收益应该用次日
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
        cum_returns = (1 + portfolio_returns).cumprod() - 1
        
        strategy_returns[name] = {
            'daily_returns': portfolio_returns,
            'cum_returns': cum_returns,
            'total_return': cum_returns.iloc[-1],
            'annual_return': (1 + cum_returns.iloc[-1]) ** (252 / len(cum_returns)) - 1,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0,
            'max_drawdown': (cum_returns.cummax() - cum_returns).max(),
            'avg_turnover': strategy.get_turnover().mean(),
        }
    
    # 打印比较结果
    print("\n策略表现对比:")
    print("-" * 80)
    print(f"{'策略名称':<25} {'总收益':>10} {'年化收益':>10} {'波动率':>10} {'夏普':>8} {'最大回撤':>10} {'换手率':>8}")
    print("-" * 80)
    
    for name, metrics in strategy_returns.items():
        print(f"{name:<25} "
              f"{metrics['total_return']:>10.2%} "
              f"{metrics['annual_return']:>10.2%} "
              f"{metrics['volatility']:>10.2%} "
              f"{metrics['sharpe']:>8.2f} "
              f"{metrics['max_drawdown']:>10.2%} "
              f"{metrics['avg_turnover']:>8.2%}")
    
    print("-" * 80)
    
    return strategy_returns


# ==================== 持仓分析 ====================

def analyze_holdings(strategy, dm, date: str = None):
    """分析策略持仓"""
    print("\n" + "="*60)
    print("持仓分析")
    print("="*60)
    
    if date is None:
        date = strategy.weights.index[-1]
    
    holdings = strategy.get_holdings(date)
    
    print(f"\n日期: {date}")
    print(f"持股数量: {len(holdings)}")
    print(f"\n持仓明细（权重从大到小）:")
    print("-" * 40)
    
    # 获取市值信息
    mcap = dm.market_cap.loc[date] if date in dm.market_cap.index else None
    
    for stock, weight in holdings.nlargest(10).items():
        if mcap is not None and stock in mcap:
            cap = mcap[stock] / 10000  # 转换为亿元
            print(f"{stock}: {weight:>8.2%}  市值: {cap:>8.1f}亿")
        else:
            print(f"{stock}: {weight:>8.2%}")
    
    if len(holdings) > 10:
        print(f"... 还有 {len(holdings) - 10} 只股票")
    
    print("-" * 40)
    print(f"总权重: {holdings.sum():.2%}")


# ==================== 主程序 ====================

def main():
    print("="*60)
    print("小市值策略演示")
    print("="*60)
    
    # 创建模拟数据
    print("\n创建模拟数据...")
    dm = create_mock_data(n_days=250, n_stocks=100)
    print(f"数据范围: {dm.price.index[0]} 至 {dm.price.index[-1]}")
    print(f"股票数量: {len(dm.price.columns)}")
    
    # 测试因子
    test_factors(dm)
    
    # 测试策略
    strategies = test_strategies(dm)
    
    # 比较策略
    compare_strategies(strategies, dm)
    
    # 分析持仓
    analyze_holdings(strategies['pure_small_cap'], dm)
    
    print("\n" + "="*60)
    print("演示完成!")
    print("="*60)


if __name__ == "__main__":
    main()
