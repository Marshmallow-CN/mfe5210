"""
量化回测系统 - 因子模块

提供多种量化因子的计算：
- 市值因子 (size): 小市值效应
- 价值因子 (value): 低估值效应  
- 动量因子 (momentum): 动量与反转效应
"""

from .base import (
    BaseFactor,
    FactorUtils,
    FactorManager,
)

from .size import (
    MarketCapFactor,
    CircMarketCapFactor,
    MarketCapRankFactor,
    SizeChangeFactor,
    SmallCapScoreFactor,
    get_size_quantile,
    filter_by_market_cap,
)

from .value import (
    PEFactor,
    PBFactor,
    PSFactor,
    DividendYieldFactor,
    ValueCompositeFactor,
    SmallValueFactor,
)

from .momentum import (
    MomentumFactor,
    ReversalFactor,
    RelativeStrengthFactor,
    VolatilityAdjustedMomentumFactor,
    MomentumChangeFactor,
    MaxDrawdownFactor,
)
from .daily_alpha import (
    ShortTermReversalFactor,
    MediumTermMomentumFactor,
    LongTermMomentumFactor,
    LowVolatilityFactor,
    DownsideVolatilityFactor,
    ReturnSkewnessFactor,
    AmihudIlliquidityFactor,
    TurnoverMeanFactor,
    TurnoverStabilityFactor,
    TurnoverShockFactor,
    PriceVolumeCorrelationFactor,
    VolumeSurgeReversalFactor,
    IntradayRangeFactor,
    CloseLocationReversalFactor,
    OvernightGapReversalFactor,
)
from .tick import (
    IntradayOFIFactor,
    IntradayVolFactor,
    IntradayLambdaFactor,
)

__all__ = [
    # 基类
    'BaseFactor',
    'FactorUtils', 
    'FactorManager',
    
    # 市值因子
    'MarketCapFactor',
    'CircMarketCapFactor',
    'MarketCapRankFactor',
    'SizeChangeFactor',
    'SmallCapScoreFactor',
    'get_size_quantile',
    'filter_by_market_cap',
    
    # 价值因子
    'PEFactor',
    'PBFactor',
    'PSFactor',
    'DividendYieldFactor',
    'ValueCompositeFactor',
    'SmallValueFactor',
    
    # 动量因子
    'MomentumFactor',
    'ReversalFactor',
    'RelativeStrengthFactor',
    'VolatilityAdjustedMomentumFactor',
    'MomentumChangeFactor',
    'MaxDrawdownFactor',
    # 作业日频alpha
    'ShortTermReversalFactor',
    'MediumTermMomentumFactor',
    'LongTermMomentumFactor',
    'LowVolatilityFactor',
    'DownsideVolatilityFactor',
    'ReturnSkewnessFactor',
    'AmihudIlliquidityFactor',
    'TurnoverMeanFactor',
    'TurnoverStabilityFactor',
    'TurnoverShockFactor',
    'PriceVolumeCorrelationFactor',
    'VolumeSurgeReversalFactor',
    'IntradayRangeFactor',
    'CloseLocationReversalFactor',
    'OvernightGapReversalFactor',
    'IntradayOFIFactor',
    'IntradayVolFactor',
    'IntradayLambdaFactor',
]
