from pathlib import Path
from typing import Optional, Union

import pandas as pd


def _parse_single_date(value: Union[str, int, float, pd.Timestamp]) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (int, float)):
        if pd.isna(value):
            raise ValueError("日期参数不能为空")
        text = str(int(value))
        if text.isdigit() and len(text) == 8:
            return pd.to_datetime(text, format="%Y%m%d", errors="raise")
        return pd.to_datetime(value, errors="raise")
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() and len(text) == 8:
            return pd.to_datetime(text, format="%Y%m%d", errors="raise")
        return pd.to_datetime(text, errors="raise")
    return pd.to_datetime(value, errors="raise")


def get_hs300_close_from_excel(
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    file_path: Optional[str] = r"d:\360MoveData\Users\Lenovo\Desktop\cuhksz\quant_backtest_claude\storage\指数数据\沪深300指数收盘价_2019至今.xlsx"
) -> pd.DataFrame:
    start_dt = _parse_single_date(start_date)
    end_dt = _parse_single_date(end_date)
    if start_dt > end_dt:
        raise ValueError("start_date 不能晚于 end_date")

    excel_path = Path(file_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"文件不存在: {excel_path}")

    df = pd.read_excel(excel_path)
    if df.empty:
        return pd.DataFrame(columns=["trade_date", "close"])

    date_col = next((c for c in ["trade_date", "交易日期", "date", "日期"] if c in df.columns), None)
    close_col = next((c for c in ["close", "收盘价", "收盘点位", "CLOSE"] if c in df.columns), None)
    if date_col is None or close_col is None:
        raise ValueError(f"未找到日期列或收盘列，当前列为: {list(df.columns)}")

    result = df[[date_col, close_col]].copy()
    result.columns = ["trade_date", "close"]
    raw_date = result["trade_date"].astype(str).str.strip()
    digit8_mask = raw_date.str.fullmatch(r"\d{8}")
    parsed_date = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
    parsed_date.loc[digit8_mask] = pd.to_datetime(raw_date.loc[digit8_mask], format="%Y%m%d", errors="coerce")
    parsed_date.loc[~digit8_mask] = pd.to_datetime(raw_date.loc[~digit8_mask], errors="coerce")
    result["trade_date"] = parsed_date
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["trade_date", "close"])
    result = result[(result["trade_date"] >= start_dt) & (result["trade_date"] <= end_dt)]
    result = result.sort_values("trade_date").reset_index(drop=True)
    return result


def get_hs300_industry_weight_from_excel(
    file_path: Optional[str] = r"d:\360MoveData\Users\Lenovo\Desktop\cuhksz\quant_backtest_claude\factor\factor_excel\沪深300行业权重_2026年3月.xlsx",
    normalize_to_ratio: bool = False
) -> pd.DataFrame:
    excel_path = Path(file_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"文件不存在: {excel_path}")

    df = pd.read_excel(excel_path)
    if df.empty:
        return pd.DataFrame(columns=["industry", "weight_pct", "constituent_count"])

    industry_col = next((c for c in ["industry", "行业", "l1_name"] if c in df.columns), None)
    weight_col = next((c for c in ["weight_pct", "weight", "行业权重"] if c in df.columns), None)
    count_col = next((c for c in ["constituent_count", "count", "成分股数量"] if c in df.columns), None)

    if industry_col is None or weight_col is None:
        raise ValueError(f"未找到行业列或权重列，当前列为: {list(df.columns)}")

    selected_cols = [industry_col, weight_col] + ([count_col] if count_col is not None else [])
    result = df[selected_cols].copy()
    rename_map = {industry_col: "industry", weight_col: "weight_pct"}
    if count_col is not None:
        rename_map[count_col] = "constituent_count"
    result = result.rename(columns=rename_map)

    result["industry"] = result["industry"].astype(str).str.strip()
    result["weight_pct"] = pd.to_numeric(result["weight_pct"], errors="coerce")
    if "constituent_count" in result.columns:
        result["constituent_count"] = pd.to_numeric(result["constituent_count"], errors="coerce")

    result = result.dropna(subset=["industry", "weight_pct"])
    result = result[result["industry"] != ""]
    result = result.groupby("industry", as_index=False).agg({
        "weight_pct": "sum",
        **({"constituent_count": "sum"} if "constituent_count" in result.columns else {})
    })
    result = result.sort_values("weight_pct", ascending=False).reset_index(drop=True)

    if normalize_to_ratio and not result.empty:
        if result["weight_pct"].max() > 1:
            result["weight_pct"] = result["weight_pct"] / 100.0

    if "constituent_count" not in result.columns:
        result["constituent_count"] = pd.NA

    return result[["industry", "weight_pct", "constituent_count"]]
