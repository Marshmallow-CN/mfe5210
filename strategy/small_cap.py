"""
量化回测系统 - 小市值策略

小市值策略是A股最经典的量化策略之一：
- 历史上A股小市值效应非常显著
- 长期来看，小市值股票组合能获得超额收益
- 但近年来随着市场成熟，小市值效应有所减弱

本模块实现多种小市值策略变体：
1. 纯小市值策略：选择市值最小的N只股票
2. 小市值+低估值：结合价值因子
3. 小市值+动量过滤：剔除下跌动量的股票
4. 小市值+反转：结合短期反转
"""
import pandas as pd
import numpy as np
from typing import Optional, TYPE_CHECKING

from .base import BaseStrategy, StrategyConfig
from factor.size import MarketCapFactor, CircMarketCapFactor, MarketCapRankFactor
from factor.value import ValueCompositeFactor
from factor.momentum import MomentumFactor, ReversalFactor
from factor.base import FactorUtils

if TYPE_CHECKING:
    from data.loader import DataManager


class SmallCapStrategy(BaseStrategy):
    """
    纯小市值策略
    
    策略逻辑：
    1. 每个调仓日，按流通市值排序
    2. 选择市值最小的N只股票
    3. 等权配置（或市值加权）
    
    适用场景：
    - 长期投资
    - 能承受较高波动
    - 相信小市值效应会持续
    
    使用示例:
        strategy = SmallCapStrategy(n_stocks=30)
        weights = strategy.run(data_manager)
    """
    
    def __init__(self, 
                 n_stocks: int = 30,
                 use_circ_mv: bool = True,
                 weight_method: str = "equal",
                 rebalance_freq: str = "weekly",
                 min_market_cap: Optional[float] = None,
                 max_market_cap: Optional[float] = 5000000,  # 默认50亿以下
                 name: str = "small_cap"):
        """
        Args:
            n_stocks: 持仓股票数量
            use_circ_mv: 是否使用流通市值（否则用总市值）
            weight_method: 权重方式 ('equal' 或 'market_cap')
            rebalance_freq: 调仓频率
            min_market_cap: 最小市值（万元），过滤壳资源股
            max_market_cap: 最大市值（万元），默认50亿
            name: 策略名称
        """
        config = StrategyConfig(
            n_stocks=n_stocks,
            weight_method=weight_method,
            rebalance_freq=rebalance_freq,
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
        )
        super().__init__(name, config)
        self.use_circ_mv = use_circ_mv
    
    def generate_signal(self, dm: "DataManager") -> pd.DataFrame:
        """
        生成小市值信号
        
        返回负市值（市值越小，信号越大）
        """
        if self.use_circ_mv:
            factor = CircMarketCapFactor(negative=True)
        else:
            factor = MarketCapFactor(negative=True)
        
        signal = factor.compute(dm)
        return signal


class SmallCapValueStrategy(BaseStrategy):
    """
    小市值+低估值策略
    
    策略逻辑：
    1. 计算市值因子得分（小市值得分高）
    2. 计算价值因子得分（低估值得分高）
    3. 综合两个因子，选择得分最高的N只股票
    
    原理：
    - 小市值提供超额收益
    - 低估值提供安全边际
    - 两者结合既有进攻性又有防御性
    """
    
    def __init__(self,
                 n_stocks: int = 30,
                 size_weight: float = 0.5,
                 value_weight: float = 0.5,
                 rebalance_freq: str = "weekly",
                 max_market_cap: Optional[float] = 5000000,
                 name: str = "small_cap_value"):
        """
        Args:
            n_stocks: 持仓股票数量
            size_weight: 市值因子权重
            value_weight: 价值因子权重
            rebalance_freq: 调仓频率
            max_market_cap: 最大市值筛选
            name: 策略名称
        """
        config = StrategyConfig(
            n_stocks=n_stocks,
            weight_method="equal",
            rebalance_freq=rebalance_freq,
            max_market_cap=max_market_cap,
        )
        super().__init__(name, config)
        self.size_weight = size_weight
        self.value_weight = value_weight
    
    def generate_signal(self, dm: "DataManager") -> pd.DataFrame:
        """生成小市值+价值信号"""
        # 市值因子
        size_factor = CircMarketCapFactor(negative=True)
        size_signal = size_factor.compute(dm)
        
        # 价值因子
        value_factor = ValueCompositeFactor()
        value_signal = value_factor.compute(dm)
        
        # 标准化
        size_std = FactorUtils.standardize(size_signal, method="rank")
        value_std = FactorUtils.standardize(value_signal, method="rank")
        
        # 加权组合
        combined = (size_std * self.size_weight + 
                    value_std * self.value_weight)
        
        return combined


class SmallCapMomentumStrategy(BaseStrategy):
    """
    小市值+动量过滤策略
    
    策略逻辑：
    1. 先按小市值选出候选池
    2. 剔除近期下跌动量的股票（避免下跌趋势）
    3. 从剩余股票中选择市值最小的N只
    
    原理：
    - 过滤掉下跌趋势的小盘股
    - 避免"接飞刀"
    - 适度牺牲小市值敞口换取风险控制
    """
    
    def __init__(self,
                 n_stocks: int = 30,
                 momentum_window: int = 20,
                 momentum_threshold: float = 0,
                 rebalance_freq: str = "weekly",
                 max_market_cap: Optional[float] = 5000000,
                 name: str = "small_cap_momentum"):
        """
        Args:
            n_stocks: 持仓股票数量
            momentum_window: 动量计算窗口
            momentum_threshold: 动量阈值，低于此值的股票被过滤
            rebalance_freq: 调仓频率
            max_market_cap: 最大市值筛选
            name: 策略名称
        """
        config = StrategyConfig(
            n_stocks=n_stocks,
            weight_method="equal",
            rebalance_freq=rebalance_freq,
            max_market_cap=max_market_cap,
        )
        super().__init__(name, config)
        self.momentum_window = momentum_window
        self.momentum_threshold = momentum_threshold
    
    def generate_signal(self, dm: "DataManager") -> pd.DataFrame:
        """生成小市值+动量过滤信号"""
        # 市值因子
        size_factor = CircMarketCapFactor(negative=True)
        size_signal = size_factor.compute(dm)
        
        # 动量因子
        momentum_factor = MomentumFactor(window=self.momentum_window)
        momentum = momentum_factor.compute(dm)
        
        # 过滤动量为负的股票
        mask = momentum < self.momentum_threshold
        size_signal = size_signal.mask(mask)
        
        return size_signal


class SmallCapReversalStrategy(BaseStrategy):
    """
    小市值+短期反转策略
    
    策略逻辑：
    1. 在小市值股票池中
    2. 选择近期下跌最多的股票（短期反转）
    3. 博弈超跌反弹
    
    原理：
    - A股小盘股短期反转效应显著
    - 超跌后容易有反弹
    - 高风险高收益策略
    
    风险提示：
    - 波动较大
    - 可能买到持续下跌的股票
    - 适合短线操作
    """
    
    def __init__(self,
                 n_stocks: int = 20,
                 reversal_window: int = 5,
                 size_weight: float = 0.5,
                 reversal_weight: float = 0.5,
                 rebalance_freq: str = "weekly",
                 max_market_cap: Optional[float] = 5000000,
                 name: str = "small_cap_reversal"):
        """
        Args:
            n_stocks: 持仓股票数量
            reversal_window: 反转计算窗口
            size_weight: 市值因子权重
            reversal_weight: 反转因子权重
            rebalance_freq: 调仓频率
            max_market_cap: 最大市值筛选
            name: 策略名称
        """
        config = StrategyConfig(
            n_stocks=n_stocks,
            weight_method="equal",
            rebalance_freq=rebalance_freq,
            max_market_cap=max_market_cap,
        )
        super().__init__(name, config)
        self.reversal_window = reversal_window
        self.size_weight = size_weight
        self.reversal_weight = reversal_weight
    
    def generate_signal(self, dm: "DataManager") -> pd.DataFrame:
        """生成小市值+反转信号"""
        # 市值因子
        size_factor = CircMarketCapFactor(negative=True)
        size_signal = size_factor.compute(dm)
        
        # 反转因子
        reversal_factor = ReversalFactor(window=self.reversal_window)
        reversal_signal = reversal_factor.compute(dm)
        
        # 标准化并组合
        size_std = FactorUtils.standardize(size_signal, method="rank")
        reversal_std = FactorUtils.standardize(reversal_signal, method="rank")
        
        combined = (size_std * self.size_weight + 
                    reversal_std * self.reversal_weight)
        
        return combined


class MicroCapStrategy(BaseStrategy):
    """
    微盘股策略
    
    专注于更小市值的股票（如10亿以下）
    高风险高收益，流动性较差
    
    适用场景：
    - 小资金量
    - 追求高收益
    - 能承受高波动和低流动性
    """
    
    def __init__(self,
                 n_stocks: int = 20,
                 max_market_cap: float = 1000000,  # 10亿以下
                 min_market_cap: float = 50000,    # 5000万以上（排除壳）
                 rebalance_freq: str = "weekly",
                 name: str = "micro_cap"):
        """
        Args:
            n_stocks: 持仓股票数量
            max_market_cap: 最大市值（万元），默认10亿
            min_market_cap: 最小市值（万元），默认5000万
            rebalance_freq: 调仓频率
            name: 策略名称
        """
        config = StrategyConfig(
            n_stocks=n_stocks,
            weight_method="equal",
            rebalance_freq=rebalance_freq,
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
        )
        super().__init__(name, config)
    
    def generate_signal(self, dm: "DataManager") -> pd.DataFrame:
        """生成微盘股信号"""
        factor = CircMarketCapFactor(negative=True)
        return factor.compute(dm)


class SmallCapRotationStrategy(BaseStrategy):
    """
    小市值轮动策略
    
    根据市场状态在不同小市值策略之间轮动：
    - 上涨市：纯小市值策略
    - 震荡市：小市值+价值策略
    - 下跌市：降低仓位或空仓
    
    使用市场动量判断市场状态
    """
    
    def __init__(self,
                 n_stocks: int = 30,
                 market_momentum_window: int = 20,
                 rebalance_freq: str = "weekly",
                 max_market_cap: Optional[float] = 5000000,
                 name: str = "small_cap_rotation"):
        """
        Args:
            n_stocks: 持仓股票数量
            market_momentum_window: 市场动量判断窗口
            rebalance_freq: 调仓频率
            max_market_cap: 最大市值筛选
            name: 策略名称
        """
        config = StrategyConfig(
            n_stocks=n_stocks,
            weight_method="equal",
            rebalance_freq=rebalance_freq,
            max_market_cap=max_market_cap,
        )
        super().__init__(name, config)
        self.market_momentum_window = market_momentum_window
    
    def generate_signal(self, dm: "DataManager") -> pd.DataFrame:
        """
        生成轮动信号
        
        根据市场状态调整因子权重
        """
        # 市值因子
        size_factor = CircMarketCapFactor(negative=True)
        size_signal = size_factor.compute(dm)
        
        # 价值因子
        value_factor = ValueCompositeFactor()
        value_signal = value_factor.compute(dm)
        
        # 计算市场动量（使用等权平均收益）
        price = dm.price
        market_return = price.pct_change(self.market_momentum_window).mean(axis=1)
        
        # 标准化因子
        size_std = FactorUtils.standardize(size_signal, method="rank")
        value_std = FactorUtils.standardize(value_signal, method="rank")
        
        # 根据市场状态调整权重
        combined = size_std.copy()
        
        for date in combined.index:
            if date not in market_return.index:
                continue
            
            mr = market_return.loc[date]
            
            if mr > 0.05:  # 上涨市
                # 纯小市值
                combined.loc[date] = size_std.loc[date]
            elif mr < -0.05:  # 下跌市
                # 更偏重价值
                combined.loc[date] = (size_std.loc[date] * 0.3 + 
                                      value_std.loc[date] * 0.7)
            else:  # 震荡市
                # 均衡
                combined.loc[date] = (size_std.loc[date] * 0.5 + 
                                      value_std.loc[date] * 0.5)
        
        return combined


# ==================== 工厂函数 ====================

def create_small_cap_strategy(variant: str = "pure", **kwargs) -> BaseStrategy:
    """
    创建小市值策略的工厂函数
    
    Args:
        variant: 策略变体
            - 'pure': 纯小市值
            - 'value': 小市值+价值
            - 'momentum': 小市值+动量过滤
            - 'reversal': 小市值+反转
            - 'micro': 微盘股
            - 'rotation': 轮动策略
        **kwargs: 传递给具体策略的参数
    
    Returns:
        对应的策略实例
    """
    strategies = {
        'pure': SmallCapStrategy,
        'value': SmallCapValueStrategy,
        'momentum': SmallCapMomentumStrategy,
        'reversal': SmallCapReversalStrategy,
        'micro': MicroCapStrategy,
        'rotation': SmallCapRotationStrategy,
    }
    
    if variant not in strategies:
        raise ValueError(f"Unknown variant: {variant}. "
                         f"Available: {list(strategies.keys())}")
    
    return strategies[variant](**kwargs)
