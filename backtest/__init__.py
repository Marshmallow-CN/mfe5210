"""
量化回测系统 - 回测模块

提供完整的回测功能：
- 回测引擎 (engine.py): 向量化回测核心
- 绩效指标 (metrics.py): 各类评价指标计算
- 回测报告 (report.py): 结果分析和可视化
"""

from .engine import (
    BacktestResult,
    BacktestConfig,
    BacktestEngine,
    SimpleBacktest,
)

from .metrics import (
    PerformanceMetrics,
    calc_all_metrics,
    calc_total_return,
    calc_annual_return,
    calc_volatility,
    calc_max_drawdown,
    calc_sharpe_ratio,
    calc_sortino_ratio,
    calc_calmar_ratio,
    calc_alpha_beta,
    calc_information_ratio,
    calc_monthly_returns,
    calc_yearly_returns,
    calc_monthly_return_table,
    calc_drawdown_series,
    calc_rolling_sharpe,
    calc_rolling_volatility,
)

from .report import (
    BacktestReport,
    compare_strategies,
    print_strategy_comparison,
)

__all__ = [
    # 引擎
    'BacktestResult',
    'BacktestConfig',
    'BacktestEngine',
    'SimpleBacktest',
    
    # 指标
    'PerformanceMetrics',
    'calc_all_metrics',
    'calc_total_return',
    'calc_annual_return',
    'calc_volatility',
    'calc_max_drawdown',
    'calc_sharpe_ratio',
    'calc_sortino_ratio',
    'calc_calmar_ratio',
    'calc_alpha_beta',
    'calc_information_ratio',
    'calc_monthly_returns',
    'calc_yearly_returns',
    'calc_monthly_return_table',
    'calc_drawdown_series',
    'calc_rolling_sharpe',
    'calc_rolling_volatility',
    
    # 报告
    'BacktestReport',
    'compare_strategies',
    'print_strategy_comparison',
]
