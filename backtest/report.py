"""
量化回测系统 - 回测报告

生成回测分析报告，包括：
- 绩效指标汇总
- 净值曲线
- 回撤分析
- 月度收益热力图
- 持仓分析
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, TYPE_CHECKING
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .engine import BacktestResult
from .metrics import (
    calc_all_metrics,
    calc_monthly_return_table,
    calc_drawdown_series,
    calc_rolling_sharpe,
    calc_rolling_volatility,
    PerformanceMetrics,
)

if TYPE_CHECKING:
    from data.loader import DataManager


class BacktestReport:
    """
    回测报告生成器
    
    使用示例:
        result = engine.run(weights, dm)
        report = BacktestReport(result)
        
        # 打印摘要
        report.print_summary()
        
        # 生成完整报告
        report.generate_report("backtest_report.html")
    """
    
    def __init__(self, 
                 result: BacktestResult,
                 strategy_name: str = "Strategy"):
        """
        初始化报告生成器
        
        Args:
            result: 回测结果
            strategy_name: 策略名称
        """
        self.result = result
        self.strategy_name = strategy_name
        
        # 计算绩效指标
        self.metrics = calc_all_metrics(
            result.portfolio_returns,
            result.benchmark_returns,
            result.turnover,
        )
    
    def print_summary(self) -> None:
        """
        打印回测摘要
        """
        print("\n" + "=" * 60)
        print(f"回测报告 - {self.strategy_name}")
        print("=" * 60)
        
        print(f"\n回测区间: {self.result.start_date} ~ {self.result.end_date}")
        print(f"交易天数: {self.metrics.trading_days}")
        
        print("\n【收益指标】")
        print(f"  累计收益率: {self.metrics.total_return:.2%}")
        print(f"  年化收益率: {self.metrics.annual_return:.2%}")
        
        print("\n【风险指标】")
        print(f"  年化波动率: {self.metrics.volatility:.2%}")
        print(f"  最大回撤:   {self.metrics.max_drawdown:.2%}")
        print(f"  95% VaR:    {self.metrics.var_95:.2%}")
        
        print("\n【风险调整收益】")
        print(f"  夏普比率:   {self.metrics.sharpe_ratio:.2f}")
        print(f"  卡尔玛比率: {self.metrics.calmar_ratio:.2f}")
        print(f"  索提诺比率: {self.metrics.sortino_ratio:.2f}")
        
        print("\n【交易指标】")
        print(f"  胜率:       {self.metrics.win_rate:.2%}")
        print(f"  盈亏比:     {self.metrics.profit_loss_ratio:.2f}")
        print(f"  平均换手率: {self.metrics.avg_turnover:.2%}")
        
        if self.metrics.alpha is not None:
            print("\n【相对指标】")
            print(f"  Alpha:      {self.metrics.alpha:.2%}")
            print(f"  Beta:       {self.metrics.beta:.2f}")
            print(f"  信息比率:   {self.metrics.information_ratio:.2f}")
            print(f"  跟踪误差:   {self.metrics.tracking_error:.2%}")
        
        print("\n" + "=" * 60)
    
    def get_summary_df(self) -> pd.DataFrame:
        """
        获取摘要DataFrame
        """
        data = {
            '指标': [
                '累计收益率', '年化收益率', '年化波动率', '最大回撤',
                '夏普比率', '卡尔玛比率', '索提诺比率',
                '胜率', '盈亏比', '平均换手率',
            ],
            '数值': [
                f"{self.metrics.total_return:.2%}",
                f"{self.metrics.annual_return:.2%}",
                f"{self.metrics.volatility:.2%}",
                f"{self.metrics.max_drawdown:.2%}",
                f"{self.metrics.sharpe_ratio:.2f}",
                f"{self.metrics.calmar_ratio:.2f}",
                f"{self.metrics.sortino_ratio:.2f}",
                f"{self.metrics.win_rate:.2%}",
                f"{self.metrics.profit_loss_ratio:.2f}",
                f"{self.metrics.avg_turnover:.2%}",
            ]
        }
        
        if self.metrics.alpha is not None:
            data['指标'].extend(['Alpha', 'Beta', '信息比率', '跟踪误差'])
            data['数值'].extend([
                f"{self.metrics.alpha:.2%}",
                f"{self.metrics.beta:.2f}",
                f"{self.metrics.information_ratio:.2f}",
                f"{self.metrics.tracking_error:.2%}",
            ])
        
        return pd.DataFrame(data)
    
    def get_monthly_returns(self) -> pd.DataFrame:
        """
        获取月度收益率表格
        """
        return calc_monthly_return_table(self.result.portfolio_returns)
    
    def get_drawdown_analysis(self) -> Dict:
        """
        获取回撤分析
        """
        drawdown = calc_drawdown_series(self.result.portfolio_returns)
        
        # 找出最大的5次回撤
        # 简化实现：找回撤的局部最小值
        dd_min = drawdown.min()
        dd_mean = drawdown.mean()
        
        return {
            '最大回撤': f"{self.metrics.max_drawdown:.2%}",
            '最大回撤持续天数': self.metrics.max_drawdown_duration,
            '平均回撤': f"{abs(dd_mean):.2%}",
            '当前回撤': f"{abs(drawdown.iloc[-1]):.2%}",
        }
    
    def get_yearly_returns(self) -> pd.Series:
        """
        获取年度收益率
        """
        yearly = self.result.portfolio_returns.resample('Y').apply(
            lambda x: (1 + x).prod() - 1
        )
        yearly.index = yearly.index.year
        return yearly
    
    def compare_with_benchmark(self) -> Optional[pd.DataFrame]:
        """
        与基准对比
        """
        if self.result.benchmark_returns is None:
            return None
        
        # 计算基准指标
        benchmark_metrics = calc_all_metrics(self.result.benchmark_returns)
        
        comparison = pd.DataFrame({
            '策略': [
                f"{self.metrics.total_return:.2%}",
                f"{self.metrics.annual_return:.2%}",
                f"{self.metrics.volatility:.2%}",
                f"{self.metrics.max_drawdown:.2%}",
                f"{self.metrics.sharpe_ratio:.2f}",
            ],
            '基准': [
                f"{benchmark_metrics.total_return:.2%}",
                f"{benchmark_metrics.annual_return:.2%}",
                f"{benchmark_metrics.volatility:.2%}",
                f"{benchmark_metrics.max_drawdown:.2%}",
                f"{benchmark_metrics.sharpe_ratio:.2f}",
            ],
            '超额': [
                f"{self.metrics.total_return - benchmark_metrics.total_return:.2%}",
                f"{self.metrics.annual_return - benchmark_metrics.annual_return:.2%}",
                f"{self.metrics.volatility - benchmark_metrics.volatility:.2%}",
                f"{benchmark_metrics.max_drawdown - self.metrics.max_drawdown:.2%}",
                f"{self.metrics.sharpe_ratio - benchmark_metrics.sharpe_ratio:.2f}",
            ]
        }, index=['累计收益', '年化收益', '波动率', '最大回撤', '夏普比率'])
        
        return comparison
    
    def generate_text_report(self) -> str:
        """
        生成文本格式报告
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"回测报告 - {self.strategy_name}")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        
        lines.append(f"\n回测区间: {self.result.start_date} ~ {self.result.end_date}")
        lines.append(f"交易天数: {self.metrics.trading_days}")
        
        lines.append("\n" + "-" * 40)
        lines.append("【绩效指标】")
        lines.append("-" * 40)
        
        for k, v in self.metrics.to_dict().items():
            lines.append(f"  {k:<15}: {v}")
        
        # 月度收益
        lines.append("\n" + "-" * 40)
        lines.append("【月度收益率】")
        lines.append("-" * 40)
        
        monthly = self.get_monthly_returns()
        lines.append(monthly.to_string())
        
        # 年度收益
        lines.append("\n" + "-" * 40)
        lines.append("【年度收益率】")
        lines.append("-" * 40)
        
        yearly = self.get_yearly_returns()
        for year, ret in yearly.items():
            lines.append(f"  {year}: {ret:.2%}")
        
        # 回撤分析
        lines.append("\n" + "-" * 40)
        lines.append("【回撤分析】")
        lines.append("-" * 40)
        
        dd_analysis = self.get_drawdown_analysis()
        for k, v in dd_analysis.items():
            lines.append(f"  {k}: {v}")
        
        # 基准对比
        if self.result.benchmark_returns is not None:
            lines.append("\n" + "-" * 40)
            lines.append("【基准对比】")
            lines.append("-" * 40)
            
            comparison = self.compare_with_benchmark()
            if comparison is not None:
                lines.append(comparison.to_string())
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def save_report(self, filepath: str) -> None:
        """
        保存报告到文件
        """
        report = self.generate_text_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"报告已保存到: {filepath}")


def compare_strategies(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """
    对比多个策略
    
    Args:
        results: 策略名称到回测结果的映射
    
    Returns:
        对比DataFrame
    """
    comparison = {}
    
    for name, result in results.items():
        metrics = calc_all_metrics(
            result.portfolio_returns,
            result.benchmark_returns,
            result.turnover,
        )
        
        comparison[name] = {
            '累计收益': f"{metrics.total_return:.2%}",
            '年化收益': f"{metrics.annual_return:.2%}",
            '波动率': f"{metrics.volatility:.2%}",
            '最大回撤': f"{metrics.max_drawdown:.2%}",
            '夏普比率': f"{metrics.sharpe_ratio:.2f}",
            '卡尔玛比率': f"{metrics.calmar_ratio:.2f}",
            '胜率': f"{metrics.win_rate:.2%}",
            '换手率': f"{metrics.avg_turnover:.2%}",
        }
    
    return pd.DataFrame(comparison).T


def print_strategy_comparison(results: Dict[str, BacktestResult]) -> None:
    """
    打印策略对比
    """
    print("\n" + "=" * 80)
    print("策略对比")
    print("=" * 80)
    
    comparison = compare_strategies(results)
    print(comparison.to_string())
    
    print("\n" + "=" * 80)
