"""
量化回测系统 - 策略基类

策略负责根据因子信号生成持仓权重矩阵 (T x N)
本模块定义策略的通用接口和辅助工具

策略工作流程：
1. 接收DataManager提供的数据
2. 计算因子信号
3. 根据信号生成目标持仓权重
4. 考虑各种约束条件（如仓位限制、换手率限制等）

关键设计：
- 策略输出权重矩阵，与回测模块解耦
- 支持多种权重计算方式（等权、市值加权、因子加权）
- 支持不同调仓频率
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, TYPE_CHECKING
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from data.loader import DataManager


@dataclass
class StrategyConfig:
    """
    策略配置参数
    """
    # 选股参数
    n_stocks: int = 50                    # 持仓股票数量
    
    # 权重方式
    weight_method: str = "equal"           # equal/market_cap/factor
    max_weight: float = 0.1               # 单只股票最大权重
    min_weight: float = 0.0               # 单只股票最小权重
    
    # 调仓参数
    rebalance_freq: str = "weekly"        # daily/weekly/monthly
    rebalance_day: int = 0                # 调仓日（周几或每月几号）
    
    # 过滤条件
    exclude_st: bool = True               # 排除ST股票
    exclude_new: bool = True              # 排除次新股
    min_list_days: int = 60               # 最小上市天数
    min_price: float = 1.0                # 最低股价
    min_volume: float = 0                 # 最小成交量（万元）
    
    # 市值范围（万元）
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None


class BaseStrategy(ABC):
    """
    策略基类
    
    所有具体策略需继承此基类，并实现 generate_signal() 方法
    
    属性:
        name: 策略名称
        config: 策略配置
        weights: 持仓权重矩阵 (T x N)
    
    使用示例:
        class MyStrategy(BaseStrategy):
            def generate_signal(self, dm):
                return dm.market_cap  # 按市值排名
        
        strategy = MyStrategy("small_cap", config)
        weights = strategy.run(data_manager)
    """
    
    def __init__(self, name: str, config: Optional[StrategyConfig] = None):
        """
        初始化策略
        
        Args:
            name: 策略名称
            config: 策略配置，默认使用StrategyConfig()
        """
        self.name = name
        self.config = config or StrategyConfig()
        self.weights: Optional[pd.DataFrame] = None
        self._signal: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def generate_signal(self, dm: "DataManager") -> pd.DataFrame:
        """
        生成选股信号（子类必须实现）
        
        Args:
            dm: DataManager实例
        
        Returns:
            信号矩阵 (T x N)，值越大代表越应该持有该股票
        """
        pass
    
    def run(self, dm: "DataManager") -> pd.DataFrame:
        """
        运行策略，生成持仓权重矩阵
        
        Args:
            dm: DataManager实例
        
        Returns:
            权重矩阵 (T x N)，每行和为1（满仓）
        """
        logger.info(f"运行策略: {self.name}")
        
        # 1. 生成原始信号
        self._signal = self.generate_signal(dm)
        
        if self._signal is None or self._signal.empty:
            logger.error("信号生成失败")
            return pd.DataFrame()
        
        # 2. 应用过滤条件
        signal = self._apply_filters(self._signal, dm)
        
        # 3. 选择调仓日
        signal = self._apply_rebalance_freq(signal)
        
        # 4. 选股并计算权重
        self.weights = self._calc_weights(signal, dm)
        
        logger.info(f"策略运行完成，生成 {len(self.weights)} 个交易日的权重")
        
        return self.weights
    
    def _apply_filters(self, signal: pd.DataFrame, dm: "DataManager") -> pd.DataFrame:
        """
        应用各种过滤条件
        """
        result = signal.copy()
        cfg = self.config
        
        # 市值过滤
        if cfg.min_market_cap is not None or cfg.max_market_cap is not None:
            mcap = dm.market_cap
            if not mcap.empty:
                if cfg.min_market_cap is not None:
                    mask = mcap < cfg.min_market_cap
                    result = result.mask(mask)
                if cfg.max_market_cap is not None:
                    mask = mcap > cfg.max_market_cap
                    result = result.mask(mask)
        
        # 价格过滤
        if cfg.min_price > 0:
            price = dm.price
            if not price.empty:
                mask = price < cfg.min_price
                result = result.mask(mask)
        
        return result
    
    def _apply_rebalance_freq(self, signal: pd.DataFrame) -> pd.DataFrame:
        """
        根据调仓频率筛选日期
        只在调仓日保留信号，非调仓日信号设为NaN
        """
        freq = self.config.rebalance_freq
        day = self.config.rebalance_day
        
        if freq == "daily":
            return signal
        
        result = signal.copy()
        dates = signal.index
        
        if freq == "weekly":
            # 获取每周的调仓日（默认周一=0）
            rebalance_dates = [d for d in dates if d.dayofweek == day]
        elif freq == "monthly":
            # 获取每月的调仓日（默认每月第一个交易日）
            monthly_groups = pd.Series(dates).groupby([dates.year, dates.month])
            if day == 0:
                rebalance_dates = [g.iloc[0] for _, g in monthly_groups]
            else:
                rebalance_dates = [g.iloc[min(day-1, len(g)-1)] for _, g in monthly_groups]
        else:
            rebalance_dates = list(dates)
        
        # 只在调仓日有信号
        non_rebalance = [d for d in dates if d not in rebalance_dates]
        result.loc[non_rebalance] = np.nan
        
        return result
    
    def _calc_weights(self, signal: pd.DataFrame, dm: "DataManager") -> pd.DataFrame:
        """
        根据信号计算持仓权重
        """
        cfg = self.config
        weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
        
        for date in signal.index:
            row = signal.loc[date].dropna()
            
            if len(row) == 0:
                continue
            
            # 选择因子值最大的N只股票
            n = min(cfg.n_stocks, len(row))
            selected = row.nlargest(n).index
            
            # 计算权重
            if cfg.weight_method == "equal":
                w = pd.Series(1.0 / n, index=selected)
            elif cfg.weight_method == "market_cap":
                mcap = dm.market_cap.loc[date, selected]
                w = mcap / mcap.sum()
            elif cfg.weight_method == "factor":
                factor_values = row.loc[selected]
                # 确保为正数
                factor_values = factor_values - factor_values.min() + 1e-10
                w = factor_values / factor_values.sum()
            else:
                w = pd.Series(1.0 / n, index=selected)
            
            # 应用权重上下限
            w = w.clip(cfg.min_weight, cfg.max_weight)
            w = w / w.sum()  # 重新归一化
            
            weights.loc[date, selected] = w
        
        # 前向填充（非调仓日保持上一期持仓）
        for i in range(1, len(weights)):
            if weights.iloc[i].sum() == 0:
                weights.iloc[i] = weights.iloc[i-1]
        
        return weights
    
    def get_holdings(self, date: str) -> pd.Series:
        """
        获取指定日期的持仓
        
        Args:
            date: 日期字符串 YYYYMMDD 或 datetime
        
        Returns:
            持仓Series，index为股票代码，values为权重
        """
        if self.weights is None:
            return pd.Series(dtype=float)
        
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        if date in self.weights.index:
            holdings = self.weights.loc[date]
            return holdings[holdings > 0]
        
        return pd.Series(dtype=float)
    
    def get_turnover(self) -> pd.Series:
        """
        计算换手率
        
        Returns:
            每日换手率 Series
        """
        if self.weights is None or self.weights.empty:
            return pd.Series(dtype=float)
        
        # 计算权重变化的绝对值之和
        weight_change = self.weights.diff().abs().sum(axis=1)
        
        # 换手率 = 权重变化 / 2 （因为买入卖出各算一次）
        turnover = weight_change / 2
        
        return turnover
    
    def summary(self) -> Dict:
        """
        策略摘要统计
        """
        if self.weights is None:
            return {}
        
        turnover = self.get_turnover()
        
        return {
            'name': self.name,
            'n_dates': len(self.weights),
            'n_stocks_avg': (self.weights > 0).sum(axis=1).mean(),
            'turnover_avg': turnover.mean(),
            'turnover_total': turnover.sum(),
            'max_weight': self.weights.max().max(),
            'config': self.config.__dict__,
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class StrategyUtils:
    """
    策略工具类
    """
    
    @staticmethod
    def combine_signals(signals: List[pd.DataFrame], 
                        weights: Optional[List[float]] = None,
                        method: str = "rank") -> pd.DataFrame:
        """
        组合多个信号
        
        Args:
            signals: 信号列表
            weights: 权重列表，默认等权
            method: 组合方法 ('rank' 或 'zscore')
        
        Returns:
            组合后的信号
        """
        if not signals:
            return pd.DataFrame()
        
        if weights is None:
            weights = [1.0 / len(signals)] * len(signals)
        
        combined = None
        for signal, w in zip(signals, weights):
            if method == "rank":
                standardized = signal.rank(axis=1, pct=True)
            else:
                # zscore
                mean = signal.mean(axis=1)
                std = signal.std(axis=1)
                standardized = signal.sub(mean, axis=0).div(std, axis=0)
            
            weighted = standardized * w
            
            if combined is None:
                combined = weighted
            else:
                combined = combined.add(weighted, fill_value=0)
        
        return combined
    
    @staticmethod
    def get_rebalance_dates(dates: pd.DatetimeIndex, 
                            freq: str = "weekly",
                            day: int = 0) -> List:
        """
        获取调仓日期列表
        
        Args:
            dates: 日期索引
            freq: 调仓频率
            day: 调仓日
        
        Returns:
            调仓日期列表
        """
        if freq == "daily":
            return list(dates)
        elif freq == "weekly":
            return [d for d in dates if d.dayofweek == day]
        elif freq == "monthly":
            monthly_groups = pd.Series(dates).groupby([dates.year, dates.month])
            if day == 0:
                return [g.iloc[0] for _, g in monthly_groups]
            else:
                return [g.iloc[min(day-1, len(g)-1)] for _, g in monthly_groups]
        return list(dates)


# ==================== 简单策略实现 ====================

class BuyAndHoldStrategy(BaseStrategy):
    """
    买入持有策略（基准策略）
    
    等权买入所有股票并持有
    """
    
    def __init__(self, name: str = "buy_and_hold"):
        config = StrategyConfig(
            n_stocks=9999,  # 持有所有股票
            weight_method="equal"
        )
        super().__init__(name, config)
    
    def generate_signal(self, dm: "DataManager") -> pd.DataFrame:
        """所有股票信号相同"""
        price = dm.price
        return pd.DataFrame(1.0, index=price.index, columns=price.columns)


class TopNStrategy(BaseStrategy):
    """
    TopN策略
    
    按某个因子排序，选择排名前N的股票
    """
    
    def __init__(self, name: str, factor: pd.DataFrame, 
                 config: Optional[StrategyConfig] = None):
        """
        Args:
            name: 策略名称
            factor: 因子矩阵
            config: 策略配置
        """
        super().__init__(name, config)
        self.factor = factor
    
    def generate_signal(self, dm: "DataManager") -> pd.DataFrame:
        """直接返回因子值作为信号"""
        return self.factor
