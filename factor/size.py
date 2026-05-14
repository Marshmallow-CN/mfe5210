"""
量化回测系统 - 市值因子

市值因子是最经典的风格因子之一：
- 小市值效应：长期来看，小市值股票往往能获得超额收益
- A股市场小市值效应尤为明显

本模块实现多种市值相关因子
"""
import pandas as pd
import numpy as np
from typing import Optional, TYPE_CHECKING

from .base import BaseFactor, FactorUtils

if TYPE_CHECKING:
    from data.loader import DataManager


class MarketCapFactor(BaseFactor):
    """
    总市值因子
    
    因子定义：股票的总市值（取对数）
    因子方向：负向（小市值获得正收益，因此我们取负数）
    
    使用示例:
        factor = MarketCapFactor()
        factor_values = factor.compute(data_manager)
    """
    
    def __init__(self, log_transform: bool = True, negative: bool = True):
        """
        Args:
            log_transform: 是否对市值取对数（推荐True，使分布更接近正态）
            negative: 是否取负数（True则小市值因子值大，便于做多小市值）
        """
        super().__init__("market_cap")
        self.log_transform = log_transform
        self.negative = negative
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """
        计算总市值因子
        
        Args:
            dm: DataManager实例
        
        Returns:
            市值因子矩阵 (T x N)
        """
        mcap = dm.market_cap.copy()
        
        if mcap.empty:
            return pd.DataFrame()
        
        # 对数变换
        if self.log_transform:
            mcap = np.log(mcap + 1)
        
        # 取负数（使小市值因子值大）
        if self.negative:
            mcap = (-mcap).round(2)
        
        self.factor_data = mcap
        return np.round(mcap, 2)


class CircMarketCapFactor(BaseFactor):
    """
    流通市值因子
    
    相比总市值，流通市值更能反映股票的实际交易特性
    对于限售股较多的股票，流通市值更有意义
    """
    
    def __init__(self, log_transform: bool = True, negative: bool = True):
        super().__init__("circ_mv")
        self.log_transform = log_transform
        self.negative = negative
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """计算流通市值因子"""
        circ_mv = dm.circ_mv.copy()
        
        if circ_mv.empty:
            return pd.DataFrame()
        
        if self.log_transform:
            circ_mv = np.log(circ_mv + 1)
        
        if self.negative:
            circ_mv = -circ_mv
        
        self.factor_data = circ_mv
        return circ_mv


class MarketCapRankFactor(BaseFactor):
    """
    市值排名因子
    
    将市值转换为每日截面排名，消除市值绝对值的影响
    排名百分比：0表示市值最小，1表示市值最大
    """
    
    def __init__(self, use_circ: bool = False, negative: bool = True):
        """
        Args:
            use_circ: 是否使用流通市值（False则使用总市值）
            negative: 是否取负数（True则小市值排名靠前）
        """
        name = "circ_mv_rank" if use_circ else "market_cap_rank"
        super().__init__(name)
        self.use_circ = use_circ
        self.negative = negative
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """计算市值排名因子"""
        if self.use_circ:
            mcap = dm.circ_mv.copy()
        else:
            mcap = dm.market_cap.copy()
        
        if mcap.empty:
            return pd.DataFrame()
        
        # 转换为排名百分比
        rank = mcap.rank(axis=1, pct=True)
        
        if self.negative:
            rank = 1 - rank  # 小市值排名高
        
        self.factor_data = rank
        return rank


class SizeChangeFactor(BaseFactor):
    """
    市值变化因子
    
    衡量股票市值在一段时间内的变化率
    可用于捕捉市值快速膨胀或收缩的股票
    """
    
    def __init__(self, window: int = 20, negative: bool = True):
        """
        Args:
            window: 计算市值变化的窗口期（交易日）
            negative: 是否取负数（True则做空市值增长快的股票）
        """
        super().__init__(f"size_change_{window}d")
        self.window = window
        self.negative = negative
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """计算市值变化因子"""
        mcap = dm.market_cap.copy()
        
        if mcap.empty:
            return pd.DataFrame()
        
        # 计算市值变化率
        change = mcap.pct_change(self.window)
        
        if self.negative:
            change = -change
        
        self.factor_data = change
        return change


class SmallCapScoreFactor(BaseFactor):
    """
    小市值综合评分因子
    
    综合考虑多个市值相关指标，给出小市值的综合评分
    - 总市值排名
    - 流通市值排名  
    - 市值变化（负向）
    
    最终评分 = 等权或加权平均
    """
    
    def __init__(self, weights: Optional[dict] = None):
        """
        Args:
            weights: 权重字典，默认等权
                     {'total_mv': 0.4, 'circ_mv': 0.4, 'change': 0.2}
        """
        super().__init__("small_cap_score")
        self.weights = weights or {
            'total_mv': 0.4,
            'circ_mv': 0.4, 
            'change': 0.2
        }
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """计算小市值综合评分"""
        total_mv = dm.market_cap
        circ_mv = dm.circ_mv
        
        if total_mv.empty or circ_mv.empty:
            return pd.DataFrame()
        
        # 计算各分项（标准化后的排名）
        # 取负对数市值并标准化
        total_mv_score = FactorUtils.standardize(
            -np.log(total_mv + 1), method="rank"
        )
        
        circ_mv_score = FactorUtils.standardize(
            -np.log(circ_mv + 1), method="rank"
        )
        
        # 市值变化（取负，偏好市值没有快速膨胀的股票）
        change = -total_mv.pct_change(20)
        change_score = FactorUtils.standardize(change, method="rank")
        
        # 加权平均
        score = (total_mv_score * self.weights['total_mv'] + 
                 circ_mv_score * self.weights['circ_mv'] +
                 change_score * self.weights['change'])
        
        self.factor_data = score
        return score


# ==================== 工具函数 ====================

def get_size_quantile(market_cap: pd.DataFrame, 
                      n_groups: int = 5) -> pd.DataFrame:
    """
    将股票按市值分组
    
    Args:
        market_cap: 市值矩阵
        n_groups: 分组数量
    
    Returns:
        分组标签矩阵，1表示最小市值组，n_groups表示最大市值组
    """
    def _quantile_label(row):
        valid = row.dropna()
        if len(valid) < n_groups:
            return pd.Series(np.nan, index=row.index)
        labels = pd.qcut(valid, n_groups, labels=range(1, n_groups + 1))
        return labels.reindex(row.index)
    
    return market_cap.apply(_quantile_label, axis=1)


def filter_by_market_cap(factor: pd.DataFrame,
                         market_cap: pd.DataFrame,
                         min_cap: Optional[float] = None,
                         max_cap: Optional[float] = None) -> pd.DataFrame:
    """
    按市值范围过滤因子
    
    Args:
        factor: 因子矩阵
        market_cap: 市值矩阵（单位：万元）
        min_cap: 最小市值（万元）
        max_cap: 最大市值（万元）
    
    Returns:
        过滤后的因子矩阵
    """
    result = factor.copy()
    
    if min_cap is not None:
        mask = market_cap < min_cap
        result = result.mask(mask)
    
    if max_cap is not None:
        mask = market_cap > max_cap
        result = result.mask(mask)
    
    return result
