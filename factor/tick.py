import pandas as pd
from typing import Optional
from .base import BaseFactor
from data.tick_loader import TickFeatureStore

class IntradayOFIFactor(BaseFactor):
    def __init__(self, name: str = "intraday_ofi"):
        super().__init__(name)
    def compute(self, dm) -> pd.DataFrame:
        store = TickFeatureStore(loader=dm.loader)
        df = store.load_daily_features(dm.daily_data["trade_date"].min().strftime("%Y%m%d"), dm.daily_data["trade_date"].max().strftime("%Y%m%d"))
        if df.empty:
            return pd.DataFrame()
        mat = df.pivot(index="ts_date", columns="ts_code", values="ofi_norm").sort_index()
        return mat

class IntradayVolFactor(BaseFactor):
    def __init__(self, name: str = "intraday_vol"):
        super().__init__(name)
    def compute(self, dm) -> pd.DataFrame:
        store = TickFeatureStore(loader=dm.loader)
        df = store.load_daily_features(dm.daily_data["trade_date"].min().strftime("%Y%m%d"), dm.daily_data["trade_date"].max().strftime("%Y%m%d"))
        if df.empty:
            return pd.DataFrame()
        mat = df.pivot(index="ts_date", columns="ts_code", values="rv_sum").sort_index()
        return mat

class IntradayLambdaFactor(BaseFactor):
    def __init__(self, name: str = "intraday_lambda"):
        super().__init__(name)
    def compute(self, dm) -> pd.DataFrame:
        store = TickFeatureStore(loader=dm.loader)
        df = store.load_daily_features(dm.daily_data["trade_date"].min().strftime("%Y%m%d"), dm.daily_data["trade_date"].max().strftime("%Y%m%d"))
        if df.empty:
            return pd.DataFrame()
        mat = df.pivot(index="ts_date", columns="ts_code", values="lambda_med").sort_index()
        return mat
