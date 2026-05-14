"""
Assignment-oriented daily alpha factors.

The factors in this module are designed for daily cross-sectional stock
selection. Each factor is oriented so that larger values imply better expected
future returns, which simplifies long-short evaluation.
"""
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

from .base import BaseFactor

if TYPE_CHECKING:
    from data.loader import DataManager


class ShortTermReversalFactor(BaseFactor):
    """Buy recent losers and short recent winners."""

    def __init__(self, window: int = 5):
        super().__init__(f"short_reversal_{window}d")
        self.window = window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        price = dm.price
        if price.empty:
            return pd.DataFrame()
        factor = -price.pct_change(self.window)
        self.factor_data = factor
        return factor


class MediumTermMomentumFactor(BaseFactor):
    """Intermediate-term momentum with an optional skip window."""

    def __init__(self, window: int = 60, skip: int = 5):
        super().__init__(f"medium_momentum_{window}d_skip_{skip}d")
        self.window = window
        self.skip = skip

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        price = dm.price
        if price.empty:
            return pd.DataFrame()
        price_end = price.shift(self.skip)
        price_start = price.shift(self.skip + self.window)
        factor = price_end / price_start - 1.0
        self.factor_data = factor
        return factor


class LongTermMomentumFactor(BaseFactor):
    """Long-horizon momentum."""

    def __init__(self, window: int = 120, skip: int = 20):
        super().__init__(f"long_momentum_{window}d_skip_{skip}d")
        self.window = window
        self.skip = skip

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        price = dm.price
        if price.empty:
            return pd.DataFrame()
        price_end = price.shift(self.skip)
        price_start = price.shift(self.skip + self.window)
        factor = price_end / price_start - 1.0
        self.factor_data = factor
        return factor


class LowVolatilityFactor(BaseFactor):
    """Prefer names with lower recent realized volatility."""

    def __init__(self, window: int = 20):
        super().__init__(f"low_volatility_{window}d")
        self.window = window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        returns = dm.price.pct_change()
        if returns.empty:
            return pd.DataFrame()
        factor = -returns.rolling(self.window).std()
        self.factor_data = factor
        return factor


class DownsideVolatilityFactor(BaseFactor):
    """Prefer names with lower downside volatility."""

    def __init__(self, window: int = 20):
        super().__init__(f"downside_volatility_{window}d")
        self.window = window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        returns = dm.price.pct_change()
        if returns.empty:
            return pd.DataFrame()
        downside = returns.where(returns < 0, 0.0)
        factor = -np.sqrt((downside.pow(2)).rolling(self.window).mean())
        self.factor_data = factor
        return factor


class ReturnSkewnessFactor(BaseFactor):
    """Higher recent return skewness can proxy for positive tail exposure."""

    def __init__(self, window: int = 20):
        super().__init__(f"return_skewness_{window}d")
        self.window = window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        returns = dm.price.pct_change()
        if returns.empty:
            return pd.DataFrame()
        factor = returns.rolling(self.window).skew()
        self.factor_data = factor
        return factor


class AmihudIlliquidityFactor(BaseFactor):
    """Daily Amihud-style illiquidity using amount traded."""

    def __init__(self, window: int = 20):
        super().__init__(f"amihud_illiquidity_{window}d")
        self.window = window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        returns = dm.price.pct_change().abs()
        amount = dm.get_matrix_from_long("日线数据", "amount")
        if returns.empty or amount.empty:
            return pd.DataFrame()
        illiq = returns / amount.replace(0, np.nan)
        factor = illiq.rolling(self.window).mean()
        self.factor_data = factor
        return factor


class TurnoverMeanFactor(BaseFactor):
    """Prefer lower turnover names."""

    def __init__(self, window: int = 20):
        super().__init__(f"turnover_mean_{window}d")
        self.window = window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        turnover = dm.get_matrix_from_long("基础指标", "turnover_rate")
        if turnover.empty:
            return pd.DataFrame()
        factor = -turnover.rolling(self.window).mean()
        self.factor_data = factor
        return factor


class TurnoverStabilityFactor(BaseFactor):
    """Prefer names with more stable turnover."""

    def __init__(self, window: int = 20):
        super().__init__(f"turnover_stability_{window}d")
        self.window = window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        turnover = dm.get_matrix_from_long("基础指标", "turnover_rate")
        if turnover.empty:
            return pd.DataFrame()
        factor = -turnover.rolling(self.window).std()
        self.factor_data = factor
        return factor


class TurnoverShockFactor(BaseFactor):
    """Short-vs-long turnover shock."""

    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__(f"turnover_shock_{short_window}_{long_window}d")
        self.short_window = short_window
        self.long_window = long_window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        turnover = dm.get_matrix_from_long("基础指标", "turnover_rate")
        if turnover.empty:
            return pd.DataFrame()
        short_mean = turnover.rolling(self.short_window).mean()
        long_mean = turnover.rolling(self.long_window).mean()
        factor = short_mean / long_mean.replace(0, np.nan) - 1.0
        self.factor_data = factor
        return factor


class PriceVolumeCorrelationFactor(BaseFactor):
    """Negative return-volume correlation as a mean-reversion signal."""

    def __init__(self, window: int = 20):
        super().__init__(f"price_volume_corr_{window}d")
        self.window = window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        returns = dm.price.pct_change()
        volume = dm.get_matrix_from_long("日线数据", "vol")
        if returns.empty or volume.empty:
            return pd.DataFrame()
        volume_change = volume.pct_change().replace([np.inf, -np.inf], np.nan)
        factor = -returns.rolling(self.window).corr(volume_change)
        self.factor_data = factor
        return factor


class VolumeSurgeReversalFactor(BaseFactor):
    """Recent sell-off combined with abnormal volume."""

    def __init__(self, return_window: int = 5, volume_window: int = 20):
        super().__init__(f"volume_surge_reversal_{return_window}_{volume_window}d")
        self.return_window = return_window
        self.volume_window = volume_window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        price = dm.price
        volume = dm.get_matrix_from_long("日线数据", "vol")
        if price.empty or volume.empty:
            return pd.DataFrame()
        recent_return = price.pct_change(self.return_window)
        volume_ratio = volume / volume.rolling(self.volume_window).mean()
        factor = -recent_return * volume_ratio
        self.factor_data = factor
        return factor


class IntradayRangeFactor(BaseFactor):
    """Prefer names with tighter recent trading ranges."""

    def __init__(self, window: int = 20):
        super().__init__(f"intraday_range_{window}d")
        self.window = window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        high = dm.get_matrix_from_long("日线数据", "high")
        low = dm.get_matrix_from_long("日线数据", "low")
        close = dm.price
        if high.empty or low.empty or close.empty:
            return pd.DataFrame()
        daily_range = (high - low) / close.replace(0, np.nan)
        factor = -daily_range.rolling(self.window).mean()
        self.factor_data = factor
        return factor


class CloseLocationReversalFactor(BaseFactor):
    """Buy names that tend to close near the intraday low."""

    def __init__(self, window: int = 10):
        super().__init__(f"close_location_reversal_{window}d")
        self.window = window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        high = dm.get_matrix_from_long("日线数据", "high")
        low = dm.get_matrix_from_long("日线数据", "low")
        close = dm.price
        if high.empty or low.empty or close.empty:
            return pd.DataFrame()
        close_loc = (close - low) / (high - low).replace(0, np.nan)
        factor = -close_loc.rolling(self.window).mean()
        self.factor_data = factor
        return factor


class OvernightGapReversalFactor(BaseFactor):
    """Negative opening gaps can mean-revert."""

    def __init__(self, window: int = 5):
        super().__init__(f"overnight_gap_reversal_{window}d")
        self.window = window

    def compute(self, dm: "DataManager") -> pd.DataFrame:
        open_price = dm.open
        pre_close = dm.get_matrix_from_long("日线数据", "pre_close")
        if open_price.empty or pre_close.empty:
            return pd.DataFrame()
        gap = open_price / pre_close.replace(0, np.nan) - 1.0
        factor = -gap.rolling(self.window).mean()
        self.factor_data = factor
        return factor
