import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Iterable, Tuple
from datetime import datetime
from .loader import TushareDataLoader
from config.settings import TICK_DATA_DIR, TICK_FEATURE_DIR

def _list_date_dirs(root: Path) -> List[Path]:
    '''列出目录下的所有日期目录，返回日期目录列表，按日期排序'''
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])

def _list_code_files(date_dir: Path, codes: Optional[Iterable[str]] = None) -> List[Path]:
    '''列出日期目录下的所有股票文件，返回股票文件列表，按股票代码排序'''
    files = sorted([p for p in date_dir.iterdir() if p.suffix.lower() == ".csv"])
    if codes:
        code_set = set(codes)
        files = [p for p in files if p.stem in code_set]
    return files

def _parse_tick_file(fp: Path, date_str: str, code: str) -> pd.DataFrame:
    names = [
        "tran_id",
        "time",
        "price",
        "volume",
        "sale_order_volume",
        "buy_order_volume",
        "type",
        "sale_order_id",
        "sale_order_price",
        "buy_order_id",
        "buy_order_price",
    ]
    df = pd.read_csv(fp, header=None, names=names, engine="python", on_bad_lines="skip")
    df["time"] = df["time"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.upper()
    numeric_cols = [
        "tran_id",
        "price",
        "volume",
        "sale_order_volume",
        "buy_order_volume",
        "sale_order_id",
        "sale_order_price",
        "buy_order_id",
        "buy_order_price",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", "price", "volume", "type"])
    df = df[(df["price"] > 0) & (df["volume"] > 0)]
    dt = pd.to_datetime(date_str + " " + df["time"], errors="coerce")
    df = df.assign(datetime=dt, code=code)
    df = df.dropna(subset=["datetime"])
    df["qty"] = df["volume"]
    df["side"] = df["type"]
    side_map = {"B": 1.0, "S": -1.0}
    df["side_val"] = df["side"].map(side_map).fillna(0.0)
    df["ofi"] = df["side_val"] * df["qty"]  # 逐笔订单流不平衡：买量记正、卖量记负
    df["logp"] = np.log(df["price"].clip(lower=1e-12))  # 价格取对数后用于收益率计算
    df["ret"] = df["logp"].diff().fillna(0.0)  # 相邻逐笔对数收益率
    df["rv_tick"] = df["ret"].pow(2)  # 逐笔实现波动分量（平方收益）
    dp = df["price"].diff().abs().fillna(0.0)
    df["lambda_tick"] = np.where(df["qty"] > 0, dp / df["qty"], 0.0)  # 逐笔价格冲击强度近似
    df["minute"] = df["datetime"].dt.floor("T")
    return df

def _aggregate_minute(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["code","minute"], sort=True)
    vol = g["qty"].sum()
    turnover = g.apply(lambda x: (x["price"] * x["qty"]).sum())
    vwap = turnover.divide(vol.replace(0, np.nan))
    ofi = g["ofi"].sum()
    rv = g["rv_tick"].sum()
    last_price = g["price"].last()
    mdf = pd.DataFrame({
        "vwap": vwap,
        "ofi": ofi,
        "rv": rv,
        "price_close": last_price,
        "volume": vol
    })
    mdf = mdf.reset_index()
    return mdf

def _aggregate_daily(minute_df: pd.DataFrame, lambda_med: float, date_str: str) -> pd.DataFrame:
    g = minute_df.groupby("code")
    total_vol = g["volume"].sum().replace(0, np.nan)
    ofi_sum = g["ofi"].sum()
    ofi_norm = ofi_sum.divide(total_vol)
    rv_sum = g["rv"].sum()
    vwap_day = g.apply(lambda x: (x["vwap"] * x["volume"]).sum() / x["volume"].sum() if x["volume"].sum() > 0 else np.nan)
    df = pd.DataFrame({
        "ts_date": pd.to_datetime(date_str),
        "code": ofi_norm.index,
        "ofi_norm": ofi_norm.values,
        "rv_sum": rv_sum.values,
        "vwap_day": vwap_day.values,
        "lambda_med": lambda_med
    })
    return df

class TickFeatureStore:
    def __init__(self, data_root: Path = TICK_DATA_DIR, feature_root: Path = TICK_FEATURE_DIR, loader: Optional[TushareDataLoader] = None):
        self.data_root = Path(data_root)
        self.feature_root = Path(feature_root)
        self.feature_file = self.feature_root / "daily_features.parquet"
        self.loader = loader or TushareDataLoader()
        stocks = self.loader.load_stock_list()
        self.symbol_to_ts = {}
        if not stocks.empty:
            self.symbol_to_ts = dict(zip(stocks["symbol"].astype(str).str.zfill(6), stocks["ts_code"]))

    def build_daily_features(self, start_date: str, end_date: str, codes: Optional[Iterable[str]] = None, overwrite: bool = False) -> Path:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        out = []
        for ddir in _list_date_dirs(self.data_root):
            try:
                d = pd.to_datetime(ddir.name)
            except Exception:
                continue
            if d < start_dt or d > end_dt:
                continue
            files = _list_code_files(ddir, codes)
            daily_concat = []
            lambda_vals = []
            for fp in files:
                code = fp.stem
                df = _parse_tick_file(fp, ddir.name, code)
                if df.empty:
                    continue
                mdf = _aggregate_minute(df)
                daily_concat.append(mdf)
                lambda_vals.append(df["lambda_tick"].median())
            if not daily_concat:
                continue
            m_all = pd.concat(daily_concat, ignore_index=True)
            lam_med = float(np.nanmedian(lambda_vals)) if lambda_vals else np.nan
            ddf = _aggregate_daily(m_all, lam_med, ddir.name)
            ddf["ts_code"] = ddf["code"].map(self.symbol_to_ts)
            ddf.dropna(subset=["ts_code"], inplace=True)
            out.append(ddf[["ts_date","ts_code","ofi_norm","rv_sum","vwap_day","lambda_med"]])
        if not out:
            return self.feature_file
        res = pd.concat(out, ignore_index=True)
        if self.feature_file.exists() and not overwrite:
            old = pd.read_parquet(self.feature_file)
            res = pd.concat([old, res], ignore_index=True).drop_duplicates(subset=["ts_date","ts_code"], keep="last")
        res.to_parquet(self.feature_file, index=False)
        return self.feature_file

    def load_daily_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.feature_file.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self.feature_file)
        df = df[(df["ts_date"] >= pd.to_datetime(start_date)) & (df["ts_date"] <= pd.to_datetime(end_date))].copy()
        return df
