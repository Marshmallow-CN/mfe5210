"""
量化回测系统 - 回测引擎

向量化回测的核心模块，负责：
1. 接收策略输出的权重矩阵
2. 计算组合收益（考虑交易成本）
3. 处理A股特殊规则（T+1、涨跌停、停牌）
4. 输出回测结果

设计要点：
- 使用向量化计算，避免逐日循环
- 权重矩阵 shift(1) 防止未来函数
- 可交易掩码处理停牌和涨跌停
- 交易成本基于权重变化计算
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from data.loader import DataManager


@dataclass
class BacktestResult:
    """
    回测结果数据类
    
    存储回测的所有输出数据
    """
    # 收益序列
    portfolio_returns: pd.Series      # 组合日收益率
    cumulative_returns: pd.Series     # 累计收益率
    net_value: pd.Series              # 净值曲线
    
    # 基准数据（如果有）
    benchmark_returns: Optional[pd.Series] = None
    benchmark_cumulative: Optional[pd.Series] = None
    benchmark_net_value: Optional[pd.Series] = None
    
    # 持仓数据
    weights: Optional[pd.DataFrame] = None      # 权重矩阵
    positions: Optional[pd.DataFrame] = None    # 实际持仓
    
    # 交易数据
    turnover: Optional[pd.Series] = None        # 换手率
    trade_costs: Optional[pd.Series] = None     # 交易成本
    
    # 元数据
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    config: Optional[dict] = None


@dataclass  
class BacktestConfig:
    """
    回测配置参数
    """
    # 交易成本
    commission_rate: float = 0.0003    # 佣金费率（双边，买卖各收一次）
    slippage: float = 0.001            # 滑点
    stamp_duty: float = 0.001          # 印花税（仅卖出收取）
    min_commission: float = 5.0        # 最低佣金（元）
    
    # 交易规则
    price_type: str = "close"          # 成交价格：close/open/vwap
    trade_delay: int = 1               # 交易延迟（1表示T+1，信号次日执行）
    
    # A股特殊规则
    apply_limit_rules: bool = True     # 是否应用涨跌停规则
    apply_suspend_rules: bool = True   # 是否应用停牌规则
    
    # 基准
    benchmark: Optional[str] = None    # 基准代码，如 "000300.SH"
    
    # 资金
    initial_capital: float = 1000000   # 初始资金


class BacktestEngine:
    """
    向量化回测引擎
    
    核心思想：
    - 权重矩阵 W[t] 表示 t 日收盘时的目标持仓
    - 实际交易发生在 t+1 日（T+1规则）
    - 收益计算：R[t+1] = W[t] * r[t+1] - cost[t]
    
    使用示例:
        engine = BacktestEngine(config)
        result = engine.run(weights, dm)
        
        print(f"总收益: {result.cumulative_returns.iloc[-1]:.2%}")
        print(f"夏普比率: {calc_sharpe(result.portfolio_returns):.2f}")
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
        """
        self.config = config or BacktestConfig()
    
    def run(self, 
            weights: pd.DataFrame, 
            dm: "DataManager",
            benchmark_returns: Optional[pd.Series] = None) -> BacktestResult:
        """
        运行回测
        
        Args:
            weights: 策略输出的权重矩阵 (T x N)，每行和为1
            dm: DataManager实例，提供价格等数据
            benchmark_returns: 基准收益率序列（可选）
        
        Returns:
            BacktestResult: 回测结果
        """
        logger.info("开始回测...")
        
        # 1. 获取价格数据
        price = dm.price
        
        # 确保权重和价格对齐
        common_dates = weights.index.intersection(price.index)
        common_stocks = weights.columns.intersection(price.columns)
        
        weights = weights.loc[common_dates, common_stocks]
        price = price.loc[common_dates, common_stocks]
        
        logger.info(f"回测区间: {common_dates[0]} ~ {common_dates[-1]}")
        logger.info(f"交易日数: {len(common_dates)}, 股票数: {len(common_stocks)}")
        
        # 2. 计算收益率矩阵
        returns = price.pct_change().fillna(0)
        
        # 3. 应用交易规则（涨跌停、停牌等）
        if self.config.apply_limit_rules or self.config.apply_suspend_rules:
            tradable_mask = self._get_tradable_mask(dm, weights.index, weights.columns)
        else:
            tradable_mask = pd.DataFrame(True, index=weights.index, columns=weights.columns)
        
        # 4. 调整权重（处理不可交易股票）
        adjusted_weights = self._adjust_weights(weights, tradable_mask)
        
        # 5. 计算组合收益（考虑交易延迟）
        # 关键：用 t-1 日的权重乘以 t 日的收益
        delayed_weights = adjusted_weights.shift(self.config.trade_delay)
        
        # 组合收益（未扣成本）
        gross_returns = (delayed_weights * returns).sum(axis=1)
        
        # 6. 计算交易成本
        turnover, trade_costs = self._calc_trading_costs(adjusted_weights)
        
        # 7. 净收益
        net_returns = gross_returns - trade_costs
        
        # 8. 计算累计收益和净值
        cumulative_returns = (1 + net_returns).cumprod() - 1
        net_value = (1 + net_returns).cumprod() * self.config.initial_capital
        
        # 9. 处理基准
        benchmark_cum = None
        benchmark_nv = None
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.reindex(net_returns.index).fillna(0)
            benchmark_cum = (1 + benchmark_returns).cumprod() - 1
            benchmark_nv = (1 + benchmark_returns).cumprod() * self.config.initial_capital
        
        logger.info(f"回测完成！总收益: {cumulative_returns.iloc[-1]:.2%}")
        
        return BacktestResult(
            portfolio_returns=net_returns,
            cumulative_returns=cumulative_returns,
            net_value=net_value,
            benchmark_returns=benchmark_returns,
            benchmark_cumulative=benchmark_cum,
            benchmark_net_value=benchmark_nv,
            weights=adjusted_weights,
            turnover=turnover,
            trade_costs=trade_costs,
            start_date=str(common_dates[0].date()),
            end_date=str(common_dates[-1].date()),
            config=self.config.__dict__,
        )
    
    def _get_tradable_mask(self, 
                           dm: "DataManager",
                           dates: pd.DatetimeIndex,
                           stocks: pd.Index) -> pd.DataFrame:
        """
        生成可交易掩码
        
        True = 可交易, False = 不可交易（停牌或涨跌停）
        """
        mask = pd.DataFrame(True, index=dates, columns=stocks)
        
        # 获取价格数据判断涨跌停
        if self.config.apply_limit_rules:
            price = dm.price.reindex(index=dates, columns=stocks)
            
            # 涨停：今日收盘价 >= 昨日收盘价 * 1.095（考虑精度）
            # 跌停：今日收盘价 <= 昨日收盘价 * 0.905
            pct_change = price.pct_change()
            
            # 涨停不能买入
            limit_up = pct_change >= 0.095
            # 跌停不能卖出（但可以买入，这里简化处理为都不交易）
            limit_down = pct_change <= -0.095
            
            limit_mask = ~(limit_up | limit_down)
            mask = mask & limit_mask
        
        # 停牌：成交量为0
        if self.config.apply_suspend_rules:
            try:
                volume = dm.get_matrix('vol', 'daily').reindex(index=dates, columns=stocks)
                suspend_mask = volume > 0
                mask = mask & suspend_mask
            except Exception:
                pass  # 如果没有成交量数据，跳过
        
        return mask
    
    def _adjust_weights(self,
                        weights: pd.DataFrame,
                        tradable_mask: pd.DataFrame) -> pd.DataFrame:
        """
        调整权重，处理不可交易股票
        
        策略：将不可交易股票的权重按比例分配给可交易股票
        """
        adjusted = weights.copy()
        
        for date in adjusted.index:
            w = adjusted.loc[date]
            mask = tradable_mask.loc[date]
            
            # 不可交易股票的权重
            untradable_weight = w[~mask].sum()
            
            if untradable_weight > 0 and mask.any():
                # 将不可交易股票的权重清零
                adjusted.loc[date, ~mask] = 0
                
                # 按比例分配给可交易股票
                tradable_weight = w[mask].sum()
                if tradable_weight > 0:
                    scale = (tradable_weight + untradable_weight) / tradable_weight
                    adjusted.loc[date, mask] = w[mask] * scale
        
        return adjusted
    
    def _calc_trading_costs(self, 
                            weights: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        计算交易成本
        
        Returns:
            turnover: 换手率序列
            costs: 交易成本序列（占组合比例）
        """
        cfg = self.config
        
        # 权重变化
        weight_change = weights.diff().abs()
        
        # 买入量和卖出量
        weight_increase = weights.diff().clip(lower=0)  # 买入
        weight_decrease = (-weights.diff()).clip(lower=0)  # 卖出
        
        # 换手率 = 权重变化总和 / 2
        turnover = weight_change.sum(axis=1) / 2
        
        # 交易成本
        # 买入成本：佣金 + 滑点
        buy_cost = weight_increase.sum(axis=1) * (cfg.commission_rate + cfg.slippage)
        
        # 卖出成本：佣金 + 滑点 + 印花税
        sell_cost = weight_decrease.sum(axis=1) * (cfg.commission_rate + cfg.slippage + cfg.stamp_duty)
        
        total_cost = buy_cost + sell_cost
        
        return turnover, total_cost

class SimpleBacktest:
    """
    简化版回测（快速测试用）
    
    不考虑复杂规则，纯向量化计算
    适合因子研究阶段的快速验证
    """
    
    @staticmethod
    def run(weights: pd.DataFrame,
            returns: pd.DataFrame,
            cost_rate: float = 0.003,
            delay: int = 1) -> pd.Series:
        """
        快速回测
        
        Args:
            weights: 权重矩阵
            returns: 收益率矩阵
            cost_rate: 总交易成本率（买卖合计）
            delay: 交易延迟天数
        
        Returns:
            净值序列
        """
        # 对齐
        common_dates = weights.index.intersection(returns.index)
        common_stocks = weights.columns.intersection(returns.columns)
        
        w = weights.loc[common_dates, common_stocks]
        r = returns.loc[common_dates, common_stocks]
        
        # 组合收益
        delayed_w = w.shift(delay)
        gross_ret = (delayed_w * r).sum(axis=1)
        
        # 交易成本
        turnover = w.diff().abs().sum(axis=1) / 2
        cost = turnover * cost_rate
        
        # 净收益
        net_ret = gross_ret - cost
        
        # 净值
        nav = (1 + net_ret).cumprod()
        
        return nav


# ==================== 辅助函数 ====================

def align_data(weights: pd.DataFrame, 
               price: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对齐权重和价格数据
    """
    common_dates = weights.index.intersection(price.index)
    common_stocks = weights.columns.intersection(price.columns)
    
    return (weights.loc[common_dates, common_stocks],
            price.loc[common_dates, common_stocks])


def calc_portfolio_returns(weights: pd.DataFrame,
                           returns: pd.DataFrame,
                           delay: int = 1) -> pd.Series:
    """
    计算组合收益（不含成本）
    """
    w, r = align_data(weights, returns)
    delayed_w = w.shift(delay)
    return (delayed_w * r).sum(axis=1)
