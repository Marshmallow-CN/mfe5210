"""
量化回测系统 - 动量因子

动量效应（Momentum）：过去表现好的股票未来继续表现好
反转效应（Reversal）：过去表现好的股票未来表现差

在不同市场和不同周期，动量和反转的有效性不同：
- A股短期（1-4周）更多呈现反转效应
- A股中期（3-12个月）动量效应较弱
- 长期有反转效应

本模块实现多种动量相关因子
"""
import pandas as pd
import numpy as np
from typing import Optional, TYPE_CHECKING

from .base import BaseFactor, FactorUtils

if TYPE_CHECKING:
    from data.loader import DataManager


class MomentumFactor(BaseFactor):
    """
    动量因子
    
    因子定义：过去N日的累计收益率
    因子方向：默认正向（做多过去涨幅大的股票）
    
    常用周期：
    - 短期动量：5-20日
    - 中期动量：60-120日（常跳过最近1个月）
    - 长期动量：250日
    """
    
    def __init__(self, window: int = 20, skip: int = 0, negative: bool = False):
        """
        Args:
            window: 计算动量的窗口期（交易日）
            skip: 跳过最近N天（避免短期反转影响）
            negative: 是否取负数（True则变成反转因子）
        """
        super().__init__(f"momentum_{window}d")
        self.window = window
        self.skip = skip
        self.negative = negative
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """
        计算动量因子
        
        动量 = (今日收盘价 - N日前收盘价) / N日前收盘价
        """
        price = dm.price
        
        if price.empty:
            return pd.DataFrame()
        
        if self.skip > 0:
            # 跳过最近skip天，计算window天前到skip天前的收益
            price_end = price.shift(self.skip)
            price_start = price.shift(self.skip + self.window)
        else:
            price_end = price
            price_start = price.shift(self.window)
        
        momentum = (price_end - price_start) / price_start
        
        if self.negative:
            momentum = -momentum
        
        self.factor_data = momentum
        return momentum


class ReversalFactor(BaseFactor):
    """
    反转因子
    
    即负动量因子，做空过去涨幅大的股票
    A股市场短期反转效应明显，通常使用5-20日窗口
    """
    
    def __init__(self, window: int = 5):
        """
        Args:
            window: 计算反转的窗口期，默认5日
        """
        super().__init__(f"reversal_{window}d")
        self.window = window
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """计算反转因子（负动量）"""
        price = dm.price
        
        if price.empty:
            return pd.DataFrame()
        
        # 负收益率
        reversal = -price.pct_change(self.window)
        
        self.factor_data = reversal
        return reversal


class RelativeStrengthFactor(BaseFactor):
    """
    相对强度因子（RSI的变体）
    
    衡量上涨天数占比和平均涨幅强度
    """
    
    def __init__(self, window: int = 20):
        super().__init__(f"relative_strength_{window}d")
        self.window = window
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """
        计算相对强度因子
        
        RS = 过去N日平均上涨幅度 / 过去N日平均下跌幅度
        """
        price = dm.price
        
        if price.empty:
            return pd.DataFrame()
        
        # 日收益率
        returns = price.pct_change()
        
        # 分离上涨和下跌
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        
        # 计算平均
        avg_gain = gains.rolling(window=self.window).mean()
        avg_loss = losses.rolling(window=self.window).mean()
        
        # 相对强度
        rs = avg_gain / (avg_loss + 1e-10)  # 避免除零
        
        self.factor_data = rs
        return rs


class VolatilityAdjustedMomentumFactor(BaseFactor):
    """
    波动率调整动量因子
    
    动量 / 波动率，衡量风险调整后的动量
    夏普比率的简化版
    """
    
    def __init__(self, window: int = 20):
        super().__init__(f"vol_adj_momentum_{window}d")
        self.window = window
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """计算波动率调整动量"""
        price = dm.price
        
        if price.empty:
            return pd.DataFrame()
        
        returns = price.pct_change()
        
        # 累计收益
        cum_return = price.pct_change(self.window)
        
        # 波动率
        volatility = returns.rolling(window=self.window).std()
        
        # 波动率调整动量
        vol_adj_mom = cum_return / (volatility + 1e-10)
        
        self.factor_data = vol_adj_mom
        return vol_adj_mom


class MomentumChangeFactor(BaseFactor):
    """
    动量变化因子（动量加速度）
    
    衡量动量的变化速度，捕捉趋势加速或减速
    """
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        """
        Args:
            short_window: 短期动量窗口
            long_window: 长期动量窗口
        """
        super().__init__(f"momentum_change_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """
        计算动量变化
        
        动量加速度 = 短期动量 - 长期动量
        """
        price = dm.price
        
        if price.empty:
            return pd.DataFrame()
        
        short_mom = price.pct_change(self.short_window)
        long_mom = price.pct_change(self.long_window)
        
        # 年化处理
        short_ann = short_mom * (252 / self.short_window)
        long_ann = long_mom * (252 / self.long_window)
        
        momentum_change = short_ann - long_ann
        
        self.factor_data = momentum_change
        return momentum_change


class MaxDrawdownFactor(BaseFactor):
    """
    最大回撤因子
    
    过去N日的最大回撤，衡量下行风险
    负向因子：回撤大的股票被低配
    """
    
    def __init__(self, window: int = 20, negative: bool = True):
        super().__init__(f"max_drawdown_{window}d")
        self.window = window
        self.negative = negative
    
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """计算最大回撤因子"""
        price = dm.price
        
        if price.empty:
            return pd.DataFrame()
        
        # 计算滚动最大回撤
        rolling_max = price.rolling(window=self.window, min_periods=1).max()
        drawdown = (price - rolling_max) / rolling_max
        max_drawdown = drawdown.rolling(window=self.window, min_periods=1).min()
        
        if self.negative:
            max_drawdown = -max_drawdown  # 转为正数，回撤大的因子值大
        
        self.factor_data = max_drawdown
        return max_drawdown
