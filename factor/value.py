"""
量化回测系统 - 价值因子

价值因子衡量股票的相对估值水平：
- PE (市盈率): 股价 / 每股收益
- PB (市净率): 股价 / 每股净资产
- PS (市销率): 股价 / 每股营收

低估值股票长期来看往往能获得超额收益（价值溢价）
"""
import pandas as pd
import numpy as np
from typing import Optional, TYPE_CHECKING

from .base import BaseFactor, FactorUtils

if TYPE_CHECKING:
    from data.loader import DataManager


class PEFactor(BaseFactor):
    """
    市盈率因子
    
    因子定义：市盈率的倒数（EP，盈利收益率）
    因子方向：正向（低PE/高EP做多）
    
    注意：PE为负的股票（亏损股）会被过滤
    """
    
    def __init__(self, use_ttm: bool = True, negative: bool = False):
        """
        Args:
            use_ttm: 是否使用TTM市盈率（滚动12个月）
            negative: 是否取负数
        """
        name = "pe_ttm" if use_ttm else "pe"
        super().__init__(name)
        self.use_ttm = use_ttm
        self.negative = negative
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """
        计算PE因子（实际返回EP）
        
        Returns:
            EP因子矩阵，即1/PE
        """
        field = 'pe_ttm' if self.use_ttm else 'pe'
        pe = dm.get_matrix_from_long('基础指标', 'pe_ttm')
        
        if pe.empty:
            return pd.DataFrame()
        
        # 过滤负PE（亏损股票）
        pe = pe.where(pe > 0)
        
        # 转换为EP（盈利收益率）
        ep = 1 / pe
        
        if self.negative:
            ep = -ep
        
        self.factor_data = ep
        return ep


class PBFactor(BaseFactor):
    """
    市净率因子
    
    因子定义：市净率的倒数（BP，账面市值比）
    因子方向：正向（低PB/高BP做多）
    """
    
    def __init__(self, negative: bool = False):
        super().__init__("pb")
        self.negative = negative
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """计算PB因子（实际返回BP）"""
        pb = dm.get_matrix_from_long('基础指标', 'pb')
        
        if pb.empty:
            return pd.DataFrame()
        
        # 过滤异常值（PB <= 0）
        pb = pb.where(pb > 0)
        
        # 转换为BP
        bp = 1 / pb
        
        if self.negative:
            bp = -bp
        
        self.factor_data = bp
        return bp


class PSFactor(BaseFactor):
    """
    市销率因子
    
    因子定义：市销率的倒数（SP，销售收益率）
    因子方向：正向（低PS/高SP做多）
    
    PS比PE更稳定，因为营收通常为正
    """
    
    def __init__(self, use_ttm: bool = True, negative: bool = False):
        name = "ps_ttm" if use_ttm else "ps"
        super().__init__(name)
        self.use_ttm = use_ttm
        self.negative = negative
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """计算PS因子（实际返回SP）"""
        field = 'ps_ttm' if self.use_ttm else 'ps'
        ps = dm.get_matrix_from_long('基础指标', 'ps_ttm')
        
        if ps.empty:
            return pd.DataFrame()
        
        # 过滤异常值
        ps = ps.where(ps > 0)
        
        # 转换为SP
        sp = 1 / ps
        
        if self.negative:
            sp = -sp
        
        self.factor_data = sp
        return sp


class DividendYieldFactor(BaseFactor):
    """
    股息率因子
    
    因子定义：近12个月股息 / 当前股价
    因子方向：正向（高股息率做多）
    
    高股息策略是经典的防御性策略
    """
    
    def __init__(self, use_ttm: bool = True):
        name = "dividend_yield_ttm" if use_ttm else "dividend_yield"
        super().__init__(name)
        self.use_ttm = use_ttm
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """计算股息率因子"""
        field = 'dv_ttm' if self.use_ttm else 'dv_ratio'
        dv = dm.get_matrix_from_long('基础指标', 'dv_ttm')
        
        if dv.empty:
            return pd.DataFrame()
        
        # 股息率直接使用，无需转换
        self.factor_data = dv
        return dv


class ValueCompositeFactor(BaseFactor):
    """
    价值综合因子
    
    综合多个价值指标，给出估值的综合评分
    默认权重：EP 40%, BP 40%, SP 20%
    """
    
    def __init__(self, weights: Optional[dict] = None):
        super().__init__("value_composite")
        self.weights = weights or {
            'ep': 0.4,
            'bp': 0.4,
            'sp': 0.2
        }
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """计算价值综合因子"""
        # 计算各单项因子
        pe = dm.get_matrix_from_long('基础指标', 'pe_ttm')
        pb = dm.get_matrix_from_long('基础指标', 'pb')
        ps = dm.get_matrix_from_long('基础指标', 'ps_ttm')
        
        if pe.empty or pb.empty or ps.empty:
            return pd.DataFrame()
        
        # 转换并标准化
        ep = 1 / pe.where(pe > 0)
        bp = 1 / pb.where(pb > 0)
        sp = 1 / ps.where(ps > 0)
        
        ep_std = FactorUtils.standardize(ep, method="rank")
        bp_std = FactorUtils.standardize(bp, method="rank")
        sp_std = FactorUtils.standardize(sp, method="rank")
        
        # 加权平均
        composite = (ep_std * self.weights['ep'] +
                     bp_std * self.weights['bp'] +
                     sp_std * self.weights['sp'])
        
        self.factor_data = composite
        return composite


class SmallValueFactor(BaseFactor):
    """
    小市值价值因子
    
    结合小市值和低估值两个维度：
    - 小市值：流通市值排名
    - 低估值：价值综合评分
    
    A股历史上，小市值+低估值是非常有效的策略组合
    """
    
    def __init__(self, size_weight: float = 0.5, value_weight: float = 0.5):
        super().__init__("small_value")
        self.size_weight = size_weight
        self.value_weight = value_weight
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """计算小市值价值因子"""
        # 市值因子（取负对数）
        mcap = dm.circ_mv
        if mcap.empty:
            return pd.DataFrame()
        
        size_score = FactorUtils.standardize(-np.log(mcap + 1), method="rank")
        
        # 价值因子
        value_factor = ValueCompositeFactor()
        value_score = value_factor.compute(dm)
        
        if value_score.empty:
            return size_score  # 如果没有价值数据，只用市值
        
        # 组合
        combined = (size_score * self.size_weight + 
                    value_score * self.value_weight)
        
        self.factor_data = combined
        return combined
