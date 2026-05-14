"""
Run the MFE5210 alpha-factor assignment workflow.

This script:
1. Loads daily data, basic data and stock metadata.
2. Computes a pool of daily cross-sectional alpha factors.
3. Applies winsorization, neutralization and standardization.
4. Evaluates each factor with a cost-free long-short portfolio.
5. Selects a low-correlation subset whose pairwise correlation stays below
   the required threshold.
6. Exports summary tables and a README-style markdown report.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.metrics import (  # noqa: E402
    calc_max_drawdown,
    calc_sharpe_ratio,
    calc_total_return,
    calc_volatility,
)
from data.loader import DataManager  # noqa: E402
from factor import (  # noqa: E402
    AmihudIlliquidityFactor,
    CircMarketCapFactor,
    CloseLocationReversalFactor,
    DividendYieldFactor,
    DownsideVolatilityFactor,
    IntradayRangeFactor,
    LongTermMomentumFactor,
    LowVolatilityFactor,
    MediumTermMomentumFactor,
    OvernightGapReversalFactor,
    PBFactor,
    PEFactor,
    PriceVolumeCorrelationFactor,
    ReturnSkewnessFactor,
    ShortTermReversalFactor,
    TurnoverMeanFactor,
    TurnoverShockFactor,
    TurnoverStabilityFactor,
    VolatilityAdjustedMomentumFactor,
    VolumeSurgeReversalFactor,
)
from factor.base import BaseFactor, FactorUtils  # noqa: E402
from factor.factor_analysis.factor_tools import (  # noqa: E402
    get_hs300_close_from_excel,
    get_hs300_industry_weight_from_excel,
)


@dataclass
class AlphaDefinition:
    name: str
    builder: Callable[[], BaseFactor]
    category: str
    reference: str


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a compact string."""
    total_seconds = max(int(seconds), 0)
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    return f"{minutes:02d}m {sec:02d}s"


def make_alpha_definitions() -> List[AlphaDefinition]:
    """Candidate alpha pool for the assignment."""
    return [
        AlphaDefinition(
            name="small_cap",
            builder=lambda: CircMarketCapFactor(negative=True),
            category="Size",
            reference="Banz (1981), Fama-French size effect",
        ),
        AlphaDefinition(
            name="ep_ttm",
            builder=lambda: PEFactor(use_ttm=True),
            category="Value",
            reference="Basu (1977) earnings yield effect",
        ),
        AlphaDefinition(
            name="bp",
            builder=lambda: PBFactor(),
            category="Value",
            reference="Rosenberg, Reid and Lanstein (1985)",
        ),
        AlphaDefinition(
            name="dividend_yield",
            builder=lambda: DividendYieldFactor(use_ttm=True),
            category="Value",
            reference="Fama-French dividend and value literature",
        ),
        AlphaDefinition(
            name="short_reversal_5d",
            builder=lambda: ShortTermReversalFactor(window=5),
            category="Reversal",
            reference="De Bondt and Thaler (1985)",
        ),
        AlphaDefinition(
            name="medium_momentum_60_5",
            builder=lambda: MediumTermMomentumFactor(window=60, skip=5),
            category="Momentum",
            reference="Jegadeesh and Titman (1993)",
        ),
        AlphaDefinition(
            name="long_momentum_120_20",
            builder=lambda: LongTermMomentumFactor(window=120, skip=20),
            category="Momentum",
            reference="Jegadeesh and Titman (1993)",
        ),
        AlphaDefinition(
            name="vol_adj_momentum_20",
            builder=lambda: VolatilityAdjustedMomentumFactor(window=20),
            category="Momentum",
            reference="Risk-adjusted momentum literature",
        ),
        AlphaDefinition(
            name="low_volatility_20",
            builder=lambda: LowVolatilityFactor(window=20),
            category="Volatility",
            reference="Ang et al. (2006), Baker and Haugen (2012)",
        ),
        AlphaDefinition(
            name="downside_volatility_20",
            builder=lambda: DownsideVolatilityFactor(window=20),
            category="Volatility",
            reference="Downside risk anomaly literature",
        ),
        AlphaDefinition(
            name="return_skewness_20",
            builder=lambda: ReturnSkewnessFactor(window=20),
            category="Risk",
            reference="Harvey and Siddique (2000)",
        ),
        AlphaDefinition(
            name="amihud_illiquidity_20",
            builder=lambda: AmihudIlliquidityFactor(window=20),
            category="Liquidity",
            reference="Amihud (2002)",
        ),
        AlphaDefinition(
            name="turnover_mean_20",
            builder=lambda: TurnoverMeanFactor(window=20),
            category="Liquidity",
            reference="Datar, Naik and Radcliffe (1998)",
        ),
        AlphaDefinition(
            name="turnover_stability_20",
            builder=lambda: TurnoverStabilityFactor(window=20),
            category="Liquidity",
            reference="Turnover persistence and attention literature",
        ),
        AlphaDefinition(
            name="turnover_shock_5_20",
            builder=lambda: TurnoverShockFactor(short_window=5, long_window=20),
            category="Liquidity",
            reference="Attention-driven trading literature",
        ),
        AlphaDefinition(
            name="price_volume_corr_20",
            builder=lambda: PriceVolumeCorrelationFactor(window=20),
            category="Price-Volume",
            reference="Price-volume relation literature",
        ),
        AlphaDefinition(
            name="volume_surge_reversal_5_20",
            builder=lambda: VolumeSurgeReversalFactor(return_window=5, volume_window=20),
            category="Price-Volume",
            reference="Short-term overreaction literature",
        ),
        AlphaDefinition(
            name="intraday_range_20",
            builder=lambda: IntradayRangeFactor(window=20),
            category="Microstructure",
            reference="Range-based volatility literature",
        ),
        AlphaDefinition(
            name="close_location_reversal_10",
            builder=lambda: CloseLocationReversalFactor(window=10),
            category="Microstructure",
            reference="Close location value / reversal literature",
        ),
        AlphaDefinition(
            name="overnight_gap_reversal_5",
            builder=lambda: OvernightGapReversalFactor(window=5),
            category="Microstructure",
            reference="Overnight return reversal literature",
        ),
    ]


def load_assignment_data(start_date: str, end_date: str) -> DataManager:
    """Load all data needed by the assignment workflow."""
    dm = DataManager()
    dm.load(start_date, end_date, "日线数据")
    dm.load(start_date, end_date, "基础指标")
    dm.load(start_date, end_date, "股票基础信息")
    return dm


def preprocess_factor(
    factor: pd.DataFrame,
    dm: DataManager,
    winsor_method: str = "mad",
    winsor_n: float = 3.0,
    standardize_method: str = "zscore",
    neutralize_industry: bool = True,
    neutralize_size: bool = True,
) -> pd.DataFrame:
    """Apply standard factor-cleaning steps."""
    if factor.empty:
        return factor

    processed = factor.copy()
    processed = processed.replace([np.inf, -np.inf], np.nan)
    processed = FactorUtils.winsorize(processed, method=winsor_method, n=winsor_n)

    if neutralize_industry:
        try:
            market_cap = dm.market_cap if neutralize_size else None
            processed = neutralize_factor_robust(processed, dm.industry, market_cap)
        except Exception:
            pass

    processed = FactorUtils.standardize(processed, method=standardize_method)
    processed = processed.replace([np.inf, -np.inf], np.nan)
    return processed


def neutralize_factor_robust(
    factor: pd.DataFrame,
    industry: pd.DataFrame,
    market_cap: Optional[pd.DataFrame] = None,
    min_common: int = 20,
) -> pd.DataFrame:
    """Neutralize factor cross-sections while safely dropping rows with NaNs."""
    result = factor.copy()

    for date in result.index:
        y = factor.loc[date].dropna()
        if len(y) < min_common:
            continue

        ind = industry.loc[date].reindex(y.index).dropna()
        common = y.index.intersection(ind.index)
        if len(common) < min_common:
            continue

        y = y.loc[common]
        x = pd.get_dummies(ind.loc[common].astype(str), prefix="ind", drop_first=True)

        if market_cap is not None:
            mcap = market_cap.loc[date].reindex(common)
            x["ln_mcap"] = np.log(mcap + 1.0)

        valid_mask = x.notna().all(axis=1) & y.notna()
        x = x.loc[valid_mask]
        y_valid = y.loc[valid_mask]
        if len(y_valid) < min_common or x.shape[1] == 0:
            continue

        model = LinearRegression()
        model.fit(x.values, y_valid.values)
        residuals = y_valid.values - model.predict(x.values)
        result.loc[date, y_valid.index] = residuals

    return result


def max_offdiag_correlation(corr_df: pd.DataFrame) -> float:
    """Maximum absolute off-diagonal correlation."""
    if corr_df.empty or corr_df.shape[1] <= 1:
        return 0.0
    values = corr_df.to_numpy(copy=True)
    np.fill_diagonal(values, np.nan)
    return float(np.nanmax(np.abs(values)))


def evaluate_factor(
    definition: AlphaDefinition,
    dm: DataManager,
    output_dir: Path,
    rebalance_freq: str,
    n_groups: int,
    min_universe: int,
    hs300_weight: pd.DataFrame,
    hs300_close: pd.DataFrame,
) -> Tuple[Optional[Dict[str, object]], Optional[pd.Series], Optional[pd.DataFrame]]:
    """Compute factor metrics and export the better-performing return series.

    Uses industry-neutral portfolio construction via FactorUtils.calc_factor_returns
    to avoid inflated returns from industry biases (e.g. simple equal-weight qcut
    producing 20%+ annualized long-only returns for illiquidity factors).

    Returns (summary, selected_returns, processed_factor).
    processed_factor is the TxN factor value matrix used for factor correlation.
    """
    factor_obj = definition.builder()
    raw_factor = factor_obj.compute(dm)
    if raw_factor.empty:
        return None, None, None

    processed_factor = preprocess_factor(raw_factor, dm)

    freq_map = {"daily": "D", "weekly": "W", "monthly": "M"}
    reb_freq = freq_map.get(str(rebalance_freq).strip().lower(), "M")

    try:
        result = FactorUtils.calc_factor_returns(
            factor=processed_factor,
            industry=dm.industry,
            close=dm.price,
            hs300_industry_weight=hs300_weight,
            hs300_index_daily=hs300_close,
            n_groups=n_groups,
            factor_positive=True,
            rebalance_freq=reb_freq,
            factor_name=definition.name,
            result_dir=str(output_dir),
            min_common=min_universe,
            plot=False,
        )
    except Exception:
        return None, None, None

    daily_returns = result["daily_returns"]
    long_only_returns = daily_returns["long_only"].dropna()
    long_short_returns = daily_returns["long_short"].dropna()

    if long_only_returns.empty and long_short_returns.empty:
        return None, None, None

    ann = result["annualized_returns"]
    annual_return_long_only = float(ann.get("long_only", np.nan))
    annual_return_long_short = float(ann.get("long_short", np.nan))

    if pd.isna(annual_return_long_only):
        annual_return_long_only = -np.inf
    if pd.isna(annual_return_long_short):
        annual_return_long_short = -np.inf

    if annual_return_long_only >= annual_return_long_short:
        selected_type = "long_only"
        selected_returns = long_only_returns
        selected_annual_return = annual_return_long_only
    else:
        selected_type = "long_short"
        selected_returns = long_short_returns
        selected_annual_return = annual_return_long_short

    volatility = calc_volatility(selected_returns)
    max_drawdown, max_drawdown_duration = calc_max_drawdown(selected_returns)
    sharpe = calc_sharpe_ratio(selected_returns, risk_free_rate=0.0)

    ic_mean = float(result.get("ic_mean", np.nan))
    ic_ir = float(result.get("ic_ir", np.nan))

    factor_output = output_dir / "factor_returns"
    factor_output.mkdir(parents=True, exist_ok=True)
    selected_returns.rename(definition.name).to_csv(factor_output / f"{definition.name}.csv", header=True)

    summary = {
        "factor_name": definition.name,
        "category": definition.category,
        "selected_portfolio": selected_type,
        "annual_return": selected_annual_return,
        "annual_return_long_only": np.nan if annual_return_long_only == -np.inf else annual_return_long_only,
        "annual_return_long_short": np.nan if annual_return_long_short == -np.inf else annual_return_long_short,
        "total_return": calc_total_return(selected_returns),
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": int(max_drawdown_duration),
        "ic_mean": ic_mean,
        "ic_ir": ic_ir,
        "n_obs": int(len(selected_returns)),
        "reference": definition.reference,
    }
    return summary, selected_returns.rename(definition.name), processed_factor


def compute_factor_spearman_correlation(
    factor_dfs: Dict[str, pd.DataFrame],
    min_stocks: int = 30,
) -> pd.DataFrame:
    """Compute pairwise Spearman rank correlation between factor value matrices.

    For each pair of factors, the cross-sectional Spearman rank correlation
    is computed at each date (using the common stock universe), then averaged
    over time. This measures whether two factors tend to rank stocks similarly,
    independent of return-series correlation which is dominated by market beta.

    Args:
        factor_dfs: dict mapping factor_name -> TxN factor value DataFrame.
        min_stocks: minimum common stocks per date to compute correlation.

    Returns:
        DataFrame with pairwise mean Spearman correlations.
    """
    names = list(factor_dfs.keys())
    n = len(names)
    corr_matrix = pd.DataFrame(np.eye(n), index=names, columns=names, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            fi = factor_dfs[names[i]]
            fj = factor_dfs[names[j]]

            common_dates = fi.index.intersection(fj.index)
            daily_corrs: List[float] = []
            for date in common_dates:
                xi = fi.loc[date].dropna()
                xj = fj.loc[date].dropna()
                common_stocks = xi.index.intersection(xj.index)
                if len(common_stocks) < min_stocks:
                    continue
                corr_val, _ = stats.spearmanr(xi.loc[common_stocks], xj.loc[common_stocks])
                if not pd.isna(corr_val):
                    daily_corrs.append(corr_val)

            if daily_corrs:
                avg_corr = float(np.mean(daily_corrs))
            else:
                avg_corr = np.nan
            corr_matrix.loc[names[i], names[j]] = avg_corr
            corr_matrix.loc[names[j], names[i]] = avg_corr

    return corr_matrix


def select_low_corr_factors(
    summary_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    max_corr: float = 0.5,
    min_annual_return: float = 0.03,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy selection sorted by Sharpe ratio, using pre-computed factor correlation.

    Args:
        summary_df: factor summary table with columns factor_name, sharpe, ic_mean, annual_return.
        corr_df: pre-computed pairwise factor correlation matrix (Spearman of factor values).
        max_corr: max allowed absolute correlation for inclusion.
        min_annual_return: minimum annual return to be considered.
    """
    selected: List[str] = []

    ranked = summary_df[summary_df["annual_return"] > min_annual_return].sort_values(
        ["sharpe", "ic_mean"], ascending=False
    )
    for factor_name in ranked["factor_name"]:
        if factor_name not in corr_df.index:
            continue
        if not selected:
            selected.append(factor_name)
            continue
        pair_corr = corr_df.loc[factor_name, selected].abs().max()
        if pd.isna(pair_corr) or pair_corr <= max_corr:
            selected.append(factor_name)

    selected_summary = ranked[ranked["factor_name"].isin(selected)].copy()
    selected_corr = corr_df.loc[selected, selected] if selected else pd.DataFrame()
    return selected_summary, selected_corr


def format_value(value: object, digits: int = 4, pct: bool = False) -> str:
    """Simple formatter for markdown export."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NaN"
    if pct:
        return f"{float(value):.{digits}%}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def dataframe_to_markdown(
    df: pd.DataFrame,
    pct_cols: Optional[List[str]] = None,
    digits: int = 4,
) -> str:
    """Render a dataframe as markdown without external dependencies."""
    if df.empty:
        return "_No data_"

    pct_cols = pct_cols or []
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            cells.append(format_value(row[col], digits=digits, pct=col in pct_cols))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def build_assignment_markdown(
    summary_df: pd.DataFrame,
    selected_summary: pd.DataFrame,
    selected_corr: pd.DataFrame,
    config: Dict[str, object],
) -> str:
    """Create a GitHub-ready README-style markdown report."""
    candidate_avg_sharpe = float(summary_df["sharpe"].mean()) if not summary_df.empty else np.nan
    selected_avg_sharpe = float(selected_summary["sharpe"].mean()) if not selected_summary.empty else np.nan
    max_corr = max_offdiag_correlation(selected_corr)

    summary_display = summary_df[
        [
            "factor_name",
            "category",
            "selected_portfolio",
            "annual_return",
            "annual_return_long_only",
            "annual_return_long_short",
            "volatility",
            "sharpe",
            "ic_mean",
            "ic_ir",
        ]
    ].sort_values("sharpe", ascending=False)
    selected_display = selected_summary[
        [
            "factor_name",
            "category",
            "selected_portfolio",
            "annual_return",
            "annual_return_long_only",
            "annual_return_long_short",
            "volatility",
            "sharpe",
            "ic_mean",
            "ic_ir",
        ]
    ].sort_values("sharpe", ascending=False)

    corr_table = selected_corr.reset_index().rename(columns={"index": "factor_name"}) if not selected_corr.empty else pd.DataFrame()
    references = selected_summary[["factor_name", "reference"]].drop_duplicates() if not selected_summary.empty else pd.DataFrame()

    lines = [
        "# MFE5210 Alpha Factors Assignment",
        "",
        "## Experiment Setup",
        "",
        f"- Universe: A-share cross-sectional equities",
        f"- Frequency: daily",
        f"- Sample: {config['start_date']} to {config['end_date']}",
        f"- Portfolio construction: within-industry grouping with HS300 industry weights (industry-neutral), keep the one with higher annual return for each factor",
        f"- Rebalance frequency: {config['rebalance_freq']}",
        f"- Number of groups: {config['n_groups']}",
        f"- Minimum universe size: {config['min_universe']}",
        f"- Cost assumption: no transaction cost",
        f"- Max allowed pairwise correlation: {config['max_corr']:.2f}",
        f"- Minimum annual return for submitted factors: {config['min_annual_return']:.2%}",
        "",
        "## Candidate Alpha Performance",
        "",
        f"- Number of candidate alphas: {len(summary_df)}",
        f"- Average Sharpe ratio of all candidate alphas: {format_value(candidate_avg_sharpe)}",
        "",
        dataframe_to_markdown(
            summary_display,
            pct_cols=["annual_return", "volatility"],
        ),
        "",
        "## Selected Low-Correlation Alpha Set",
        "",
        f"- Number of selected alphas: {len(selected_summary)}",
        f"- Average Sharpe ratio of selected alphas: {format_value(selected_avg_sharpe)}",
        f"- Max pairwise correlation within selected set: {format_value(max_corr)}",
        f"- Selection rule: annual return > {config['min_annual_return']:.2%} and pairwise Spearman correlation of factor values <= {config['max_corr']:.2f}",
        "",
        dataframe_to_markdown(
            selected_display,
            pct_cols=["annual_return", "annual_return_long_only", "annual_return_long_short", "volatility"],
        ),
        "",
        "## Correlation Matrix Of Selected Alphas (Spearman of factor values)",
        "",
        dataframe_to_markdown(corr_table, digits=3),
        "",
        "## References",
        "",
        dataframe_to_markdown(references, digits=3),
        "",
    ]
    return "\n".join(lines)


def run_alpha_suite(
    start_date: str = "20200101",
    end_date: str = "20251219",
    rebalance_freq: str = "monthly",
    n_groups: int = 5,
    min_universe: int = 80,
    max_corr: float = 0.5,
    min_annual_return: float = 0.03,
) -> Dict[str, object]:
    """Main workflow used by the course assignment."""
    suite_start = time.time()
    output_dir = PROJECT_ROOT / "factor" / "factor_result" / "mfe5210_alpha_suite"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading data: {start_date} -> {end_date}")
    dm = load_assignment_data(start_date, end_date)
    hs300_weight = get_hs300_industry_weight_from_excel()
    hs300_close = get_hs300_close_from_excel(start_date, end_date)
    definitions = make_alpha_definitions()
    total_factors = len(definitions)
    print(f"[2/3] Evaluating {total_factors} candidate factors...")

    summaries: List[Dict[str, object]] = []
    factor_returns: List[pd.Series] = []
    factor_values: Dict[str, pd.DataFrame] = {}
    factor_durations: List[float] = []

    for idx, definition in enumerate(definitions, start=1):
        factor_start = time.time()
        elapsed = time.time() - suite_start
        eta = "calculating..."
        if factor_durations:
            avg_factor_time = sum(factor_durations) / len(factor_durations)
            eta = format_elapsed(avg_factor_time * (total_factors - idx + 1))
        print(
            f"  [{idx:02d}/{total_factors:02d}] {definition.name} | "
            f"elapsed {format_elapsed(elapsed)} | eta {eta}"
        )

        summary, returns, factor_df = evaluate_factor(
            definition=definition,
            dm=dm,
            output_dir=output_dir,
            rebalance_freq=rebalance_freq,
            n_groups=n_groups,
            min_universe=min_universe,
            hs300_weight=hs300_weight,
            hs300_close=hs300_close,
        )

        factor_cost = time.time() - factor_start
        factor_durations.append(factor_cost)
        if summary is None or returns is None:
            print(f"      skipped | cost {format_elapsed(factor_cost)}")
            continue

        summaries.append(summary)
        factor_returns.append(returns)
        factor_values[definition.name] = factor_df
        print(
            f"      done | {summary['selected_portfolio']} | "
            f"annual {summary['annual_return']:.2%} | sharpe {summary['sharpe']:.4f} | "
            f"cost {format_elapsed(factor_cost)}"
        )

    summary_df = pd.DataFrame(summaries).sort_values("sharpe", ascending=False).reset_index(drop=True)
    returns_df = pd.concat(factor_returns, axis=1).sort_index() if factor_returns else pd.DataFrame()
    # Compute pairwise Spearman correlation on factor value matrices (not on returns)
    factor_corr_df = compute_factor_spearman_correlation(factor_values)
    selected_summary, selected_corr = select_low_corr_factors(
        summary_df,
        factor_corr_df,
        max_corr=max_corr,
        min_annual_return=min_annual_return,
    )

    config = {
        "start_date": start_date,
        "end_date": end_date,
        "rebalance_freq": rebalance_freq,
        "n_groups": n_groups,
        "min_universe": min_universe,
        "max_corr": max_corr,
        "min_annual_return": min_annual_return,
    }

    print("[3/3] Writing output files...")
    summary_df.to_csv(output_dir / "alpha_summary.csv", index=False, encoding="utf-8-sig")
    returns_df.to_csv(output_dir / "alpha_long_short_returns.csv", encoding="utf-8-sig")
    selected_summary.to_csv(output_dir / "selected_alpha_summary.csv", index=False, encoding="utf-8-sig")
    selected_corr.to_csv(output_dir / "selected_alpha_correlation.csv", encoding="utf-8-sig")

    markdown = build_assignment_markdown(summary_df, selected_summary, selected_corr, config)
    (output_dir / "README_assignment.md").write_text(markdown, encoding="utf-8")
    print(f"Finished in {format_elapsed(time.time() - suite_start)}")

    return {
        "summary": summary_df,
        "returns": returns_df,
        "selected_summary": selected_summary,
        "selected_corr": selected_corr,
        "output_dir": output_dir,
        "readme_path": output_dir / "README_assignment.md",
    }


def main() -> None:
    result = run_alpha_suite()
    print("=" * 80)
    print("MFE5210 alpha suite finished")
    print("=" * 80)
    print(f"Output directory: {result['output_dir']}")
    print(f"README summary:   {result['readme_path']}")
    if not result["selected_summary"].empty:
        print("\nSelected factors:")
        print(result["selected_summary"][["factor_name", "sharpe", "ic_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
