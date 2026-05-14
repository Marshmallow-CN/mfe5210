import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from data.tick_loader import TickFeatureStore
from data.loader import TushareDataLoader, DataManager
from factor.tick import IntradayOFIFactor, IntradayVolFactor, IntradayLambdaFactor
from factor.base import FactorUtils

DATE_START = "20250103"
DATE_END = "20250121"
# 为加速首次验证，限定部分股票；后续可置为 None 处理全量
CODES = ["000001","000002","000004","000006","000007","000008","000009","000010"]

def main():
    store = TickFeatureStore(loader=TushareDataLoader())
    fp = store.build_daily_features(DATE_START, DATE_END, codes=CODES)
    print(f"tick features saved: {fp}")

    dm = DataManager(loader=store.loader).load(DATE_START, DATE_END)
    returns = dm.price.pct_change().shift(-1)

    ofi = IntradayOFIFactor().compute(dm)
    vol = IntradayVolFactor().compute(dm)
    lam = IntradayLambdaFactor().compute(dm)

    returns = returns.reindex(ofi.index)

    ic_ofi = FactorUtils.calc_ic(ofi, returns, min_common=5)
    ic_vol = FactorUtils.calc_ic(vol, returns, min_common=5)
    ic_lam = FactorUtils.calc_ic(lam, returns, min_common=5)

    print("IC summary:")
    print("OFI:", ic_ofi.describe())
    print("VOL:", ic_vol.describe())
    print("LAM:", ic_lam.describe())

    grp_ofi = FactorUtils.calc_factor_returns(ofi, returns, n_groups=5, min_common=5)
    grp_vol = FactorUtils.calc_factor_returns(vol, returns, n_groups=5, min_common=5)
    grp_lam = FactorUtils.calc_factor_returns(lam, returns, n_groups=5, min_common=5)

    print("Layered returns (OFI):")
    print(grp_ofi.cumsum().tail())
    print("Layered returns (VOL):")
    print(grp_vol.cumsum().tail())
    print("Layered returns (LAM):")
    print(grp_lam.cumsum().tail())

if __name__ == "__main__":
    main()
