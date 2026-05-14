"""
量化回测系统 - 策略模块

提供多种量化交易策略：
- 小市值策略 (small_cap): 经典的A股量化策略
- 更多策略待添加...
"""

from .base import (
    StrategyConfig,
    BaseStrategy,
    StrategyUtils,
    BuyAndHoldStrategy,
    TopNStrategy,
)

from .small_cap import (
    SmallCapStrategy,
    SmallCapValueStrategy,
    SmallCapMomentumStrategy,
    SmallCapReversalStrategy,
    MicroCapStrategy,
    SmallCapRotationStrategy,
    create_small_cap_strategy,
)

__all__ = [
    # 基类
    'StrategyConfig',
    'BaseStrategy',
    'StrategyUtils',
    'BuyAndHoldStrategy',
    'TopNStrategy',
    
    # 小市值策略
    'SmallCapStrategy',
    'SmallCapValueStrategy',
    'SmallCapMomentumStrategy',
    'SmallCapReversalStrategy',
    'MicroCapStrategy',
    'SmallCapRotationStrategy',
    'create_small_cap_strategy',
]
