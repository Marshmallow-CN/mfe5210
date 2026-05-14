"""
量化回测系统 - 绩效指标

计算策略的各类评价指标：
- 收益指标：年化收益、累计收益
- 风险指标：波动率、最大回撤、VaR
- 风险调整收益：夏普比率、卡尔玛比率、索提诺比率
- 相对指标：Alpha、Beta、信息比率

所有指标计算均使用向量化方法
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Union, Tuple
from dataclasses import dataclass
from scipy import stats


# ==================== 数据类 ====================

@dataclass
class PerformanceMetrics:
    """
    绩效指标汇总
    """
    # 收益指标
    total_return: float          # 累计收益率
    annual_return: float         # 年化收益率
    
    # 风险指标
    volatility: float            # 年化波动率
    max_drawdown: float          # 最大回撤
    max_drawdown_duration: int   # 最大回撤持续天数
    var_95: float                # 95% VaR
    cvar_95: float               # 95% CVaR (ES)
    
    # 风险调整收益
    sharpe_ratio: float          # 夏普比率
    calmar_ratio: float          # 卡尔玛比率
    sortino_ratio: float         # 索提诺比率
    
    # 交易指标
    win_rate: float              # 胜率
    profit_loss_ratio: float     # 盈亏比
    avg_turnover: float          # 平均换手率
    
    # 相对指标（如果有基准）
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    
    # 其他
    trading_days: int = 0
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            '累计收益率': f"{self.total_return:.2%}",
            '年化收益率': f"{self.annual_return:.2%}",
            '年化波动率': f"{self.volatility:.2%}",
            '最大回撤': f"{self.max_drawdown:.2%}",
            '最大回撤天数': self.max_drawdown_duration,
            '95% VaR': f"{self.var_95:.2%}",
            '95% CVaR': f"{self.cvar_95:.2%}",
            '夏普比率': f"{self.sharpe_ratio:.2f}",
            '卡尔玛比率': f"{self.calmar_ratio:.2f}",
            '索提诺比率': f"{self.sortino_ratio:.2f}",
            '胜率': f"{self.win_rate:.2%}",
            '盈亏比': f"{self.profit_loss_ratio:.2f}",
            '平均换手率': f"{self.avg_turnover:.2%}",
            'Alpha': f"{self.alpha:.2%}" if self.alpha else 'N/A',
            'Beta': f"{self.beta:.2f}" if self.beta else 'N/A',
            '信息比率': f"{self.information_ratio:.2f}" if self.information_ratio else 'N/A',
            '跟踪误差': f"{self.tracking_error:.2%}" if self.tracking_error else 'N/A',
            '交易天数': self.trading_days,
        }
    
    def __repr__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.to_dict().items()])


# ==================== 收益指标 ====================

def calc_total_return(returns: pd.Series) -> float:
    """
    计算累计收益率
    
    Args:
        returns: 日收益率序列
    
    Returns:
        累计收益率
    """
    return (1 + returns).prod() - 1


def calc_annual_return(returns: pd.Series, 
                       periods_per_year: int = 252) -> float:
    """
    计算年化收益率
    
    Args:
        returns: 日收益率序列
        periods_per_year: 每年交易日数
    
    Returns:
        年化收益率
    """
    total_return = calc_total_return(returns)
    n_periods = len(returns)
    
    if n_periods == 0:
        return 0.0
    
    # 年化公式：(1 + total_return) ^ (periods_per_year / n_periods) - 1
    return (1 + total_return) ** (periods_per_year / n_periods) - 1


def calc_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    计算累计收益率序列
    """
    return (1 + returns).cumprod() - 1


def calc_net_value(returns: pd.Series, 
                   initial_value: float = 1.0) -> pd.Series:
    """
    计算净值曲线
    """
    return (1 + returns).cumprod() * initial_value


# ==================== 风险指标 ====================

def calc_volatility(returns: pd.Series,
                    periods_per_year: int = 252) -> float:
    """
    计算年化波动率
    """
    return returns.std() * np.sqrt(periods_per_year)


def calc_downside_volatility(returns: pd.Series,
                             threshold: float = 0,
                             periods_per_year: int = 252) -> float:
    """
    计算下行波动率（只考虑负收益）
    """
    downside_returns = returns[returns < threshold]
    if len(downside_returns) == 0:
        return 0.0
    return downside_returns.std() * np.sqrt(periods_per_year)


def calc_max_drawdown(returns: pd.Series) -> Tuple[float, int]:
    """
    计算最大回撤和最大回撤持续天数
    
    Returns:
        (最大回撤, 最大回撤持续天数)
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    
    max_dd = drawdown.min()
    
    # 计算最大回撤持续天数
    # 找到回撤开始和结束的位置
    dd_end_idx = drawdown.idxmin()
    dd_start_idx = cum_returns[:dd_end_idx].idxmax()
    
    # 找到恢复点（如果有的话）
    after_trough = cum_returns[dd_end_idx:]
    recovery_mask = after_trough >= cum_returns[dd_start_idx]
    
    if recovery_mask.any():
        dd_recovery_idx = recovery_mask.idxmax()
        duration = len(returns.loc[dd_start_idx:dd_recovery_idx])
    else:
        duration = len(returns.loc[dd_start_idx:])
    
    return abs(max_dd), duration


def calc_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    计算回撤序列
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    return (cum_returns - running_max) / running_max


def calc_var(returns: pd.Series, 
             confidence: float = 0.95) -> float:
    """
    计算VaR (Value at Risk)
    
    Args:
        returns: 收益率序列
        confidence: 置信度
    
    Returns:
        VaR值（正数表示损失）
    """
    return -np.percentile(returns, (1 - confidence) * 100)


def calc_cvar(returns: pd.Series,
              confidence: float = 0.95) -> float:
    """
    计算CVaR (Conditional VaR / Expected Shortfall)
    """
    var = calc_var(returns, confidence)
    return -returns[returns <= -var].mean()


# ==================== 风险调整收益 ====================

def calc_sharpe_ratio(returns: pd.Series,
                      risk_free_rate: float = 0.03,
                      periods_per_year: int = 252) -> float:
    """
    计算夏普比率
    
    Sharpe = (年化收益 - 无风险利率) / 年化波动率
    """
    annual_return = calc_annual_return(returns, periods_per_year)
    volatility = calc_volatility(returns, periods_per_year)
    
    if volatility == 0:
        return 0.0
    
    return (annual_return - risk_free_rate) / volatility


def calc_sortino_ratio(returns: pd.Series,
                       risk_free_rate: float = 0.03,
                       periods_per_year: int = 252) -> float:
    """
    计算索提诺比率
    
    Sortino = (年化收益 - 无风险利率) / 下行波动率
    """
    annual_return = calc_annual_return(returns, periods_per_year)
    downside_vol = calc_downside_volatility(returns, 0, periods_per_year)
    
    if downside_vol == 0:
        return 0.0
    
    return (annual_return - risk_free_rate) / downside_vol


def calc_calmar_ratio(returns: pd.Series,
                      periods_per_year: int = 252) -> float:
    """
    计算卡尔玛比率
    
    Calmar = 年化收益 / 最大回撤
    """
    annual_return = calc_annual_return(returns, periods_per_year)
    max_dd, _ = calc_max_drawdown(returns)
    
    if max_dd == 0:
        return 0.0
    
    return annual_return / max_dd


def calc_information_ratio(returns: pd.Series,
                           benchmark_returns: pd.Series,
                           periods_per_year: int = 252) -> float:
    """
    计算信息比率
    
    IR = 超额收益 / 跟踪误差
    """
    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    annual_excess = calc_annual_return(excess_returns, periods_per_year)
    return annual_excess / tracking_error


def calc_tracking_error(returns: pd.Series,
                        benchmark_returns: pd.Series,
                        periods_per_year: int = 252) -> float:
    """
    计算跟踪误差
    """
    excess_returns = returns - benchmark_returns
    return excess_returns.std() * np.sqrt(periods_per_year)


# ==================== Alpha/Beta分析 ====================

def calc_alpha_beta(returns: pd.Series,
                    benchmark_returns: pd.Series,
                    risk_free_rate: float = 0.03,
                    periods_per_year: int = 252) -> Tuple[float, float]:
    """
    计算Alpha和Beta
    
    使用CAPM模型：R_p - R_f = Alpha + Beta * (R_m - R_f) + epsilon
    
    Returns:
        (alpha, beta)
    """
    # 日化无风险利率
    rf_daily = risk_free_rate / periods_per_year
    
    # 超额收益
    excess_returns = returns - rf_daily
    excess_benchmark = benchmark_returns - rf_daily
    
    # 线性回归
    # Beta = Cov(R_p, R_m) / Var(R_m)
    cov = excess_returns.cov(excess_benchmark)
    var = excess_benchmark.var()
    
    if var == 0:
        return 0.0, 0.0
    
    beta = cov / var
    
    # Alpha = E(R_p) - Beta * E(R_m)  (已减去无风险利率)
    alpha_daily = excess_returns.mean() - beta * excess_benchmark.mean()
    alpha_annual = alpha_daily * periods_per_year
    
    return alpha_annual, beta


# ==================== 交易指标 ====================

def calc_win_rate(returns: pd.Series) -> float:
    """
    计算胜率（正收益天数占比）
    """
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


def calc_profit_loss_ratio(returns: pd.Series) -> float:
    """
    计算盈亏比（平均盈利 / 平均亏损）
    """
    profits = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(losses) == 0 or losses.mean() == 0:
        return np.inf if len(profits) > 0 else 0.0
    
    return abs(profits.mean() / losses.mean())


def calc_avg_turnover(turnover: pd.Series) -> float:
    """
    计算平均换手率
    """
    return turnover.mean()


# ==================== 综合计算 ====================

def calc_all_metrics(returns: pd.Series,
                     benchmark_returns: Optional[pd.Series] = None,
                     turnover: Optional[pd.Series] = None,
                     risk_free_rate: float = 0.03,
                     periods_per_year: int = 252) -> PerformanceMetrics:
    """
    计算所有绩效指标
    
    Args:
        returns: 策略日收益率序列
        benchmark_returns: 基准日收益率序列（可选）
        turnover: 换手率序列（可选）
        risk_free_rate: 无风险利率
        periods_per_year: 每年交易日数
    
    Returns:
        PerformanceMetrics: 绩效指标汇总
    """
    # 基础收益指标
    total_return = calc_total_return(returns)
    annual_return = calc_annual_return(returns, periods_per_year)
    
    # 风险指标
    volatility = calc_volatility(returns, periods_per_year)
    max_dd, max_dd_duration = calc_max_drawdown(returns)
    var_95 = calc_var(returns, 0.95)
    cvar_95 = calc_cvar(returns, 0.95)
    
    # 风险调整收益
    sharpe = calc_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calc_calmar_ratio(returns, periods_per_year)
    sortino = calc_sortino_ratio(returns, risk_free_rate, periods_per_year)
    
    # 交易指标
    win_rate = calc_win_rate(returns)
    pl_ratio = calc_profit_loss_ratio(returns)
    avg_turnover = calc_avg_turnover(turnover) if turnover is not None else 0.0
    
    # 相对指标
    alpha = beta = ir = te = None
    if benchmark_returns is not None:
        # 对齐数据
        common_idx = returns.index.intersection(benchmark_returns.index)
        r = returns.loc[common_idx]
        b = benchmark_returns.loc[common_idx]
        
        alpha, beta = calc_alpha_beta(r, b, risk_free_rate, periods_per_year)
        ir = calc_information_ratio(r, b, periods_per_year)
        te = calc_tracking_error(r, b, periods_per_year)
    
    return PerformanceMetrics(
        total_return=total_return,
        annual_return=annual_return,
        volatility=volatility,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        var_95=var_95,
        cvar_95=cvar_95,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        sortino_ratio=sortino,
        win_rate=win_rate,
        profit_loss_ratio=pl_ratio,
        avg_turnover=avg_turnover,
        alpha=alpha,
        beta=beta,
        information_ratio=ir,
        tracking_error=te,
        trading_days=len(returns),
    )


# ==================== 滚动指标 ====================

def calc_rolling_sharpe(returns: pd.Series,
                        window: int = 252,
                        risk_free_rate: float = 0.03) -> pd.Series:
    """
    计算滚动夏普比率
    """
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    
    return (rolling_mean - risk_free_rate) / rolling_std


def calc_rolling_max_drawdown(returns: pd.Series,
                              window: int = 252) -> pd.Series:
    """
    计算滚动最大回撤
    """
    cum_returns = (1 + returns).cumprod()
    
    def _max_dd(x):
        cum = (1 + x).cumprod()
        return (cum / cum.cummax() - 1).min()
    
    return returns.rolling(window).apply(_max_dd, raw=False)


def calc_rolling_volatility(returns: pd.Series,
                            window: int = 20) -> pd.Series:
    """
    计算滚动波动率
    """
    return returns.rolling(window).std() * np.sqrt(252)


# ==================== 月度/年度统计 ====================

def calc_monthly_returns(returns: pd.Series) -> pd.Series:
    """
    计算月度收益率
    """
    return returns.resample('M').apply(lambda x: (1 + x).prod() - 1)


def calc_yearly_returns(returns: pd.Series) -> pd.Series:
    """
    计算年度收益率
    """
    return returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)


def calc_monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """
    生成月度收益率表格
    
    Returns:
        DataFrame，行为年份，列为月份
    """
    monthly = calc_monthly_returns(returns)
    
    # 创建表格
    table = pd.DataFrame()
    for date, ret in monthly.items():
        year = date.year
        month = date.month
        table.loc[year, month] = ret
    
    # 添加年度汇总
    yearly = calc_yearly_returns(returns)
    for date, ret in yearly.items():
        table.loc[date.year, 'Year'] = ret
    
    table.columns = [f'M{i}' if i != 'Year' else 'Year' for i in table.columns]
    
    return table
