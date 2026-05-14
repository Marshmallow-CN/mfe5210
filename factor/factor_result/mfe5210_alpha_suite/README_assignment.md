# MFE5210 Alpha Factors Assignment

## Experiment Setup

- Universe: A-share cross-sectional equities
- Frequency: daily
- Sample: 20250101 to 20251219
- Portfolio construction: within-industry grouping with HS300 industry weights (industry-neutral), keep the one with higher annual return for each factor
- Rebalance frequency: monthly
- Number of groups: 5
- Minimum universe size: 80
- Cost assumption: no transaction cost
- Max allowed pairwise correlation: 0.50
- Minimum annual return for submitted factors: 3.00%

## Candidate Alpha Performance

- Number of candidate alphas: 20
- Average Sharpe ratio of all candidate alphas: 1.7662

| factor_name | category | selected_portfolio | annual_return | annual_return_long_only | annual_return_long_short | volatility | sharpe | ic_mean | ic_ir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| turnover_mean_20 | Liquidity | long_only | 40.0721% | 0.4007 | 0.0852 | 17.4631% | 2.2947 | 0.0549 | 0.5863 |
| medium_momentum_60_5 | Momentum | long_only | 35.6526% | 0.3565 | -0.0132 | 16.3514% | 2.1804 | -0.0279 | -0.3291 |
| downside_volatility_20 | Volatility | long_only | 37.3164% | 0.3732 | 0.0688 | 17.2651% | 2.1614 | 0.0380 | 0.3310 |
| turnover_stability_20 | Liquidity | long_only | 38.5134% | 0.3851 | 0.0964 | 18.1368% | 2.1235 | 0.0622 | 0.7372 |
| low_volatility_20 | Volatility | long_only | 35.8194% | 0.3582 | 0.0661 | 17.3137% | 2.0688 | 0.0574 | 0.5336 |
| bp | Value | long_only | 34.1951% | 0.3420 | 0.0147 | 17.6334% | 1.9392 | 0.0304 | 0.3827 |
| ep_ttm | Value | long_only | 32.9370% | 0.3294 | 0.0195 | 17.7155% | 1.8592 | 0.0229 | 0.2423 |
| amihud_illiquidity_20 | Liquidity | long_only | 31.5031% | 0.3150 | 0.0263 | 17.3328% | 1.8175 | 0.0177 | 0.2543 |
| price_volume_corr_20 | Price-Volume | long_only | 31.5796% | 0.3158 | 0.0186 | 17.4508% | 1.8096 | 0.0396 | 0.7366 |
| dividend_yield | Value | long_only | 29.6637% | 0.2966 | -0.0077 | 16.5951% | 1.7875 | 0.0185 | 0.2079 |
| close_location_reversal_10 | Microstructure | long_only | 36.2253% | 0.3623 | 0.0252 | 20.4152% | 1.7744 | 0.0132 | 0.1727 |
| intraday_range_20 | Microstructure | long_only | 30.3933% | 0.3039 | 0.0413 | 17.1507% | 1.7721 | 0.0613 | 0.5222 |
| short_reversal_5d | Reversal | long_only | 34.2826% | 0.3428 | 0.0087 | 20.0810% | 1.7072 | 0.0469 | 0.5886 |
| return_skewness_20 | Risk | long_only | 32.6136% | 0.3261 | 0.0277 | 19.3433% | 1.6860 | -0.0267 | -0.5983 |
| vol_adj_momentum_20 | Momentum | long_only | 28.7728% | 0.2877 | 0.0018 | 18.5894% | 1.5478 | -0.0390 | -0.5106 |
| small_cap | Size | long_only | 29.8354% | 0.2984 | 0.0082 | 19.2815% | 1.5474 | -0.0024 | -0.0421 |
| overnight_gap_reversal_5 | Microstructure | long_only | 28.8576% | 0.2886 | -0.0394 | 19.9935% | 1.4433 | -0.0157 | -0.3091 |
| long_momentum_120_20 | Momentum | long_only | 23.7761% | 0.2378 | -0.0289 | 17.4560% | 1.3621 | -0.0129 | -0.1619 |
| volume_surge_reversal_5_20 | Price-Volume | long_only | 24.1869% | 0.2419 | -0.0026 | 19.5217% | 1.2390 | 0.0483 | 0.5941 |
| turnover_shock_5_20 | Liquidity | long_only | 22.2726% | 0.2227 | -0.0316 | 18.5224% | 1.2025 | -0.0401 | -0.6163 |

## Selected Low-Correlation Alpha Set

- Number of selected alphas: 13
- Average Sharpe ratio of selected alphas: 1.7296
- Max pairwise correlation within selected set: 0.4431
- Selection rule: annual return > 3.00% and pairwise Spearman correlation of factor values <= 0.50

| factor_name | category | selected_portfolio | annual_return | annual_return_long_only | annual_return_long_short | volatility | sharpe | ic_mean | ic_ir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| turnover_mean_20 | Liquidity | long_only | 40.0721% | 40.0721% | 8.5230% | 17.4631% | 2.2947 | 0.0549 | 0.5863 |
| medium_momentum_60_5 | Momentum | long_only | 35.6526% | 35.6526% | -1.3158% | 16.3514% | 2.1804 | -0.0279 | -0.3291 |
| bp | Value | long_only | 34.1951% | 34.1951% | 1.4703% | 17.6334% | 1.9392 | 0.0304 | 0.3827 |
| ep_ttm | Value | long_only | 32.9370% | 32.9370% | 1.9524% | 17.7155% | 1.8592 | 0.0229 | 0.2423 |
| amihud_illiquidity_20 | Liquidity | long_only | 31.5031% | 31.5031% | 2.6252% | 17.3328% | 1.8175 | 0.0177 | 0.2543 |
| price_volume_corr_20 | Price-Volume | long_only | 31.5796% | 31.5796% | 1.8590% | 17.4508% | 1.8096 | 0.0396 | 0.7366 |
| close_location_reversal_10 | Microstructure | long_only | 36.2253% | 36.2253% | 2.5194% | 20.4152% | 1.7744 | 0.0132 | 0.1727 |
| short_reversal_5d | Reversal | long_only | 34.2826% | 34.2826% | 0.8651% | 20.0810% | 1.7072 | 0.0469 | 0.5886 |
| vol_adj_momentum_20 | Momentum | long_only | 28.7728% | 28.7728% | 0.1799% | 18.5894% | 1.5478 | -0.0390 | -0.5106 |
| small_cap | Size | long_only | 29.8354% | 29.8354% | 0.8213% | 19.2815% | 1.5474 | -0.0024 | -0.0421 |
| overnight_gap_reversal_5 | Microstructure | long_only | 28.8576% | 28.8576% | -3.9366% | 19.9935% | 1.4433 | -0.0157 | -0.3091 |
| long_momentum_120_20 | Momentum | long_only | 23.7761% | 23.7761% | -2.8890% | 17.4560% | 1.3621 | -0.0129 | -0.1619 |
| turnover_shock_5_20 | Liquidity | long_only | 22.2726% | 22.2726% | -3.1578% | 18.5224% | 1.2025 | -0.0401 | -0.6163 |

## Correlation Matrix Of Selected Alphas (Spearman of factor values)

| factor_name | turnover_mean_20 | medium_momentum_60_5 | bp | ep_ttm | amihud_illiquidity_20 | price_volume_corr_20 | close_location_reversal_10 | short_reversal_5d | vol_adj_momentum_20 | small_cap | overnight_gap_reversal_5 | long_momentum_120_20 | turnover_shock_5_20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| turnover_mean_20 | 1.000 | -0.346 | 0.295 | 0.252 | 0.415 | 0.269 | -0.113 | -0.014 | -0.136 | -0.301 | -0.172 | -0.267 | 0.078 |
| medium_momentum_60_5 | -0.346 | 1.000 | -0.123 | -0.134 | -0.077 | -0.142 | 0.000 | 0.059 | 0.309 | -0.020 | 0.093 | 0.443 | -0.039 |
| bp | 0.295 | -0.123 | 1.000 | 0.384 | 0.004 | 0.085 | -0.102 | 0.022 | -0.039 | -0.078 | -0.058 | -0.183 | 0.003 |
| ep_ttm | 0.252 | -0.134 | 0.384 | 1.000 | 0.044 | 0.088 | -0.056 | 0.027 | -0.057 | -0.040 | -0.061 | -0.187 | 0.002 |
| amihud_illiquidity_20 | 0.415 | -0.077 | 0.004 | 0.044 | 1.000 | 0.152 | -0.053 | -0.070 | 0.045 | 0.206 | -0.158 | -0.051 | 0.115 |
| price_volume_corr_20 | 0.269 | -0.142 | 0.085 | 0.088 | 0.152 | 1.000 | -0.019 | 0.140 | -0.408 | -0.030 | -0.078 | 0.017 | -0.088 |
| close_location_reversal_10 | -0.113 | 0.000 | -0.102 | -0.056 | -0.053 | -0.019 | 1.000 | 0.350 | -0.315 | 0.022 | -0.077 | 0.065 | -0.227 |
| short_reversal_5d | -0.014 | 0.059 | 0.022 | 0.027 | -0.070 | 0.140 | 0.350 | 1.000 | -0.403 | 0.003 | 0.225 | 0.019 | -0.389 |
| vol_adj_momentum_20 | -0.136 | 0.309 | -0.039 | -0.057 | 0.045 | -0.408 | -0.315 | -0.403 | 1.000 | -0.008 | -0.044 | -0.032 | 0.322 |
| small_cap | -0.301 | -0.020 | -0.078 | -0.040 | 0.206 | -0.030 | 0.022 | 0.003 | -0.008 | 1.000 | -0.003 | -0.034 | -0.010 |
| overnight_gap_reversal_5 | -0.172 | 0.093 | -0.058 | -0.061 | -0.158 | -0.078 | -0.077 | 0.225 | -0.044 | -0.003 | 1.000 | 0.042 | -0.057 |
| long_momentum_120_20 | -0.267 | 0.443 | -0.183 | -0.187 | -0.051 | 0.017 | 0.065 | 0.019 | -0.032 | -0.034 | 0.042 | 1.000 | -0.093 |
| turnover_shock_5_20 | 0.078 | -0.039 | 0.003 | 0.002 | 0.115 | -0.088 | -0.227 | -0.389 | 0.322 | -0.010 | -0.057 | -0.093 | 1.000 |

## References

| factor_name | reference |
| --- | --- |
| turnover_mean_20 | Datar, Naik and Radcliffe (1998) |
| medium_momentum_60_5 | Jegadeesh and Titman (1993) |
| bp | Rosenberg, Reid and Lanstein (1985) |
| ep_ttm | Basu (1977) earnings yield effect |
| amihud_illiquidity_20 | Amihud (2002) |
| price_volume_corr_20 | Price-volume relation literature |
| close_location_reversal_10 | Close location value / reversal literature |
| short_reversal_5d | De Bondt and Thaler (1985) |
| vol_adj_momentum_20 | Risk-adjusted momentum literature |
| small_cap | Banz (1981), Fama-French size effect |
| overnight_gap_reversal_5 | Overnight return reversal literature |
| long_momentum_120_20 | Jegadeesh and Titman (1993) |
| turnover_shock_5_20 | Attention-driven trading literature |
