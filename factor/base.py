"""
量化回测系统 - 因子基类

因子是量化投资的核心概念，代表某种可能预测股票收益的特征。
本模块定义因子的基类和通用工具方法。

设计思路：
- 因子以矩阵形式存储 (T x N)，T为交易日数，N为股票数
- 支持因子标准化、中性化、去极值等预处理
- 支持IC/IR等因子评价指标计算
"""
import pandas as pd
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING, Dict, Any, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 类型提示，避免循环导入
if TYPE_CHECKING:
    from data.loader import DataManager

class BaseFactor(ABC):
    """
    因子基类
    
    所有具体因子类需继承此基类，并实现 compute() 方法
    
    属性:
        name: 因子名称
        factor_data: 因子值矩阵 (T x N)
    
    使用示例:
        class MyFactor(BaseFactor):
            def compute(self, dm):
                return dm.price.pct_change(20)  # 20日收益率因子
        
        factor = MyFactor("momentum_20")
        factor_values = factor.compute(data_manager)
    """
    
    def __init__(self, name: str):
        """
        初始化因子
        
        Args:
            name: 因子名称，用于标识和日志
        """
        self.name = name
        self.factor_data: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def compute(self, dm: "DataManager") -> pd.DataFrame:
        """
        计算因子值（子类必须实现）
        
        Args:
            dm: DataManager实例，提供数据访问
                - dm.price: 收盘价矩阵
                - dm.market_cap: 总市值矩阵
                - dm.circ_mv: 流通市值矩阵
                - dm.get_matrix(field, source): 获取任意字段矩阵
        
        Returns:
            因子值矩阵 (T x N)，index为trade_date，columns为ts_code
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class FactorUtils:
    """
    因子处理工具类
    
    提供因子预处理和评价的静态方法
    """
    
    @staticmethod
    def winsorize(factor: pd.DataFrame, 
                  method: str = "mad", 
                  n: float = 3.0) -> pd.DataFrame:
        """
        去极值（缩尾处理）
        
        Args:
            factor: 因子矩阵 (T x N)
            method: 去极值方法
                - 'mad': 中位数绝对偏差法 (推荐)
                - 'std': 标准差法
                - 'percentile': 百分位数法
            n: 阈值倍数 (对于mad和std) 或百分位数 (对于percentile, 如0.01表示1%和99%)
        
        Returns:
            处理后的因子矩阵
        """
        result = factor.copy()
        
        for date in result.index:
            row = result.loc[date].dropna()
            if len(row) == 0:
                continue
            
            if method == "mad":
                # MAD: Median Absolute Deviation
                median = row.median()
                mad = np.abs(row - median).median()
                lower = median - n * 1.4826 * mad
                upper = median + n * 1.4826 * mad
            elif method == "std":
                mean = row.mean()
                std = row.std()
                lower = mean - n * std
                upper = mean + n * std
            elif method == "percentile":
                lower = row.quantile(n)
                upper = row.quantile(1 - n)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            result.loc[date] = result.loc[date].clip(lower, upper)
        
        return result
    
    @staticmethod
    def standardize(factor: pd.DataFrame, 
                    method: str = "zscore") -> pd.DataFrame:
        """
        因子标准化（截面标准化）
        
        Args:
            factor: 因子矩阵 (T x N)
            method: 标准化方法
                - 'zscore': Z-score标准化 (均值0，标准差1)
                - 'rank': 排名标准化 (0-1之间)
                - 'minmax': 最小最大标准化 (0-1之间)
        
        Returns:
            标准化后的因子矩阵
        """
        result = factor.copy()
        
        if method == "zscore":
            # 按行（截面）进行z-score标准化
            result = result.sub(result.mean(axis=1), axis=0)
            result = result.div(result.std(axis=1), axis=0)
        elif method == "rank":
            # 按行进行排名标准化
            result = result.rank(axis=1, pct=True)
        elif method == "minmax":
            # 按行进行最小最大标准化
            min_val = result.min(axis=1)
            max_val = result.max(axis=1)
            result = result.sub(min_val, axis=0).div(max_val - min_val, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return result
    
    @staticmethod
    def neutralize(factor: pd.DataFrame, 
                   industry: pd.DataFrame,
                   market_cap: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        因子中性化（去除行业和市值(可选)影响）
        
        使用线性回归的残差作为中性化后的因子值
        
        Args:
            factor: 因子矩阵 (T x N)
            industry: 行业矩阵 (T x N)，值为行业代码
            market_cap: 市值矩阵 (T x N)，可选，用于市值中性化
        
        Returns:
            中性化后的因子矩阵
        """
        result = factor.copy()
        
        for date in result.index:
            # 获取当日因子值Series
            y = factor.loc[date].dropna()
            # 获取当日股票列表
            stocks = y.index
            
            # 构建行业哑变量
            ind = industry.loc[date].reindex(stocks).dropna() #这行代码的用处是挑选出所有在stocks中的
            #股票，并把不在stocks中的股票的行业代码设为NaN并drop掉。最后ind是当日有因子值又有股票值的series
            common = ind.index #因为ind.index一定是stocks的子集
        
            if len(common) < 10:
                continue
            
            y = y.loc[common]
            ind_dummies = pd.get_dummies(ind.loc[common], prefix='ind', drop_first=True)#这行是把每个行业名称做one-hot编码
            #变成行业哑变量矩阵
            
            # 添加市值因子
            X = ind_dummies.copy()
            #如果进行市值中性化，添加市值因子进入矩阵
            if market_cap is not None:
                mcap = market_cap.loc[date].reindex(common)
                X['ln_mcap'] = np.log(mcap + 1)
            
            # 回归取残差
            try:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X.values, y.values)
                residuals = y.values - model.predict(X.values)
                result.loc[date, common] = residuals
            except Exception as e:
                print(f"行业中性化报错 {date}: {e}")
        
        return result
    
    @staticmethod
    def calc_ic(factor: pd.DataFrame, 
                dm: "DataManager",
                method: str = "spearman",
                lag: int = 1,
                ) -> pd.DataFrame:
        """
        计算因子IC (Information Coefficient)
        
        IC = 因子值与未来收益的相关系数
        
        Args:
            factor: 因子矩阵 (T x N)
            returns: 收益率矩阵 (T x N)
            method: 相关系数方法 ('spearman' 或 'pearson')
            lag: 滞后期数,默认1表示用今天的因子预测明天的收益
        
        Returns:
            每个截面日期的IC值 Series
        """
        ic_series = []
        
        #计算股票的往后lag天收益率矩阵
        def calculate_return(price,lag):
            '''
            计算股票的往后lag天收益率矩阵
            '''
            return_data = price.shift(-lag)/price -1
            return return_data
        
        price = dm.price.copy()
        returns = calculate_return(price,lag)
        dates = factor.index.intersection(returns.index)#取因子和收益的交集日期

        for date in dates:
            factor_values = factor.loc[date].dropna()
            return_values = returns.loc[date].dropna()
            
            # 取交集,取的是股票代码的交集，确保因子值和收益都有对应的股票
            common = factor_values.index.intersection(return_values.index)
            if len(common) < 30:
                continue #跳过这一天，不计算这一天的IC，因为股票数量不足30个，无法计算相关系数
            x = factor_values.loc[common]
            y = return_values.loc[common]
            
            if method == "spearman":
                ic, _ = stats.spearmanr(x, y)
            else:
                ic, _ = stats.pearsonr(x, y)
            
            ic_series.append({'date': date, 'ic': ic})
        
        if not ic_series:
            return pd.Series(dtype=float)
        
        return pd.DataFrame(ic_series).set_index('date')
    
    @staticmethod
    def calc_factor_returns(factor: pd.DataFrame,
                            industry: Optional[pd.DataFrame] = None,
                            close: Optional[pd.DataFrame] = None,
                            hs300_industry_weight: Optional[pd.DataFrame] = None,
                            hs300_index_daily: Optional[pd.DataFrame] = None,
                            n_groups: int = 5,
                            factor_positive: bool = True,
                            rebalance_freq: str = "D",
                            factor_name: str = "因子",
                            result_dir: str = r"d:\360MoveData\Users\Lenovo\Desktop\cuhksz\quant_backtest_claude\factor\factor_result",
                            risk_free_rate_path: str = r"d:\360MoveData\Users\Lenovo\Desktop\cuhksz\quant_backtest_claude\storage\宏观数据\中国十年期国债收益率历史数据.csv",
                            ic_lags: Optional[List[int]] = None,
                            weight: str = "equal",
                            min_common: Optional[int] = None,
                            plot: bool = True) -> Any:
        """
        计算因子分层组合收益。

        输出行业权重约束下的多空组合、多头组合、基准组合净值与年化收益率，并绘图。
        factor_positive=True表示因子值越大预期收益越高；
        factor_positive=False表示因子值越小预期收益越高（如市值因子）。
        rebalance_freq支持"D"(日频)、"W"(周频)、"M"(月频)调仓。
        支持IC/IR与IC衰减分析，ic_lags默认[1, 5, 10, 20, 40]。
        结果会输出Markdown报告到result_dir目录，文件名为factor_name.md。
        """
        required_inputs = [industry, close, hs300_industry_weight, hs300_index_daily]
        if any(x is None for x in required_inputs):
            raise ValueError("请传入industry、close、hs300_industry_weight、hs300_index_daily以计算行业中性组合。")

        if n_groups < 2:
            raise ValueError("n_groups至少为2。")

        # 统一调仓频率的输入写法，兼容大小写和别名
        freq_map = {
            'D': 'D',
            'DAY': 'D',
            'DAILY': 'D',
            'W': 'W',
            'WEEK': 'W',
            'WEEKLY': 'W',
            'M': 'M',
            'MONTH': 'M',
            'MONTHLY': 'M'
        }
        freq_key = str(rebalance_freq).strip().upper()
        if freq_key not in freq_map:
            raise ValueError("rebalance_freq仅支持'D'、'W'、'M'。")
        rebalance_freq_norm = freq_map[freq_key]

        # 统一解析日期索引，支持YYYYMMDD与常见日期格式
        def _to_datetime_index(idx: pd.Index) -> pd.DatetimeIndex:
            raw = pd.Series(idx.astype(str)).str.strip()
            digit8_mask = raw.str.fullmatch(r"\d{8}")
            parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
            if digit8_mask.any():
                parsed.loc[digit8_mask] = pd.to_datetime(raw.loc[digit8_mask], format="%Y%m%d", errors="coerce")
            if (~digit8_mask).any():
                parsed.loc[~digit8_mask] = pd.to_datetime(raw.loc[~digit8_mask], errors="coerce")
            if parsed.isna().any():
                raise ValueError("日期索引存在无法解析的值，请统一为YYYYMMDD或标准日期格式。")
            return pd.DatetimeIndex(parsed.values)

        factor = factor.copy()
        factor.index = _to_datetime_index(factor.index)
        factor = factor.sort_index()
        # 将行业与价格矩阵对齐到因子矩阵的日期和股票池
        industry = industry.reindex(index=factor.index, columns=factor.columns)
        close = close.reindex(index=factor.index, columns=factor.columns)

        # 采用t到t+1的前瞻收益作为持有期收益
        forward_returns = close.shift(-1) / close - 1
        if ic_lags is None:
            ic_lag_list = [1, 5, 10, 20, 40]
        else:
            ic_lag_list = sorted({int(x) for x in ic_lags if int(x) > 0})
            if len(ic_lag_list) == 0:
                ic_lag_list = [1, 5, 10, 20, 40]
        main_ic_lag = 1 if 1 in ic_lag_list else ic_lag_list[0]

        def _calc_ic_series(lag: int) -> pd.Series:
            lag_returns = close.shift(-lag) / close - 1
            values = []
            idx = []
            for d in factor.index:
                x = factor.loc[d].dropna()
                y = lag_returns.loc[d].dropna()
                common = x.index.intersection(y.index)
                if len(common) < 30:
                    continue
                ic_val, _ = stats.spearmanr(x.loc[common], y.loc[common])
                if pd.isna(ic_val):
                    continue
                idx.append(d)
                values.append(float(ic_val))
            return pd.Series(values, index=pd.DatetimeIndex(idx), name=f"IC_lag_{lag}", dtype=float)

        # 解析沪深300行业权重，兼容Series/DataFrame及不同列名
        if isinstance(hs300_industry_weight, pd.Series):
            ind_weight_series = hs300_industry_weight.copy()
            ind_weight_series.index = ind_weight_series.index.astype(str)
            ind_weight_series = ind_weight_series.astype(float)
        else:
            weight_df = hs300_industry_weight.copy()
            industry_col = None
            for c in ['industry', '行业', 'l1_name']:
                if c in weight_df.columns:
                    industry_col = c
                    break
            if industry_col is None:
                industry_col = weight_df.columns[0]

            weight_col = None
            for c in ['weight_pct', 'weight', '行业权重']:
                if c in weight_df.columns:
                    weight_col = c
                    break
            if weight_col is None:
                numeric_cols = weight_df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    raise ValueError("hs300_industry_weight未找到可用权重列。")
                weight_col = numeric_cols[0]

            ind_weight_series = weight_df.set_index(industry_col)[weight_col].astype(float)

        if ind_weight_series.max() > 1.0:
            ind_weight_series = ind_weight_series / 100.0
        ind_weight_series = ind_weight_series[ind_weight_series > 0]

        # 构造基准净值（优先使用close，若无则使用pct_chg）
        benchmark_df = hs300_index_daily.copy()
        benchmark_date_col = next((c for c in ['trade_date', '交易日期', 'date', '日期'] if c in benchmark_df.columns), None)
        if benchmark_date_col is not None:
            benchmark_df[benchmark_date_col] = _to_datetime_index(pd.Index(benchmark_df[benchmark_date_col]))
            benchmark_df = benchmark_df.set_index(benchmark_date_col)
        benchmark_df.index = _to_datetime_index(pd.Index(benchmark_df.index))
        benchmark_df = benchmark_df.sort_index()
        benchmark_close_col = None
        for c in ['close', 'CLOSE', '收盘价']:
            if c in benchmark_df.columns:
                benchmark_close_col = c
                break
        if benchmark_close_col is not None:
            benchmark_close = pd.to_numeric(benchmark_df[benchmark_close_col], errors='coerce').dropna()
            if benchmark_close.empty:
                raise ValueError("hs300_index_daily的close列为空。")
            benchmark_nav_full = benchmark_close / benchmark_close.iloc[0]
        elif 'pct_chg' in benchmark_df.columns:
            benchmark_ret_raw = pd.to_numeric(benchmark_df['pct_chg'], errors='coerce') / 100.0
            benchmark_nav_full = (1 + benchmark_ret_raw.fillna(0.0)).cumprod()
        else:
            raise ValueError("hs300_index_daily需要包含close或pct_chg列。")

        group_rets = {f'G{i+1}': [] for i in range(n_groups)}
        long_short_rets = []
        ret_dates = []
        holding_period_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        threshold = n_groups * 10 if min_common is None else min_common
        all_dates = factor.index
        # 根据频率确定“每个调仓周期的首个交易日”作为换仓日
        if rebalance_freq_norm == 'D':
            rebalance_dates = all_dates[:-1]
        elif rebalance_freq_norm == 'W':
            rebalance_dates = pd.Series(all_dates, index=all_dates).groupby(all_dates.to_period('W')).first().values
        else:
            rebalance_dates = pd.Series(all_dates, index=all_dates).groupby(all_dates.to_period('M')).first().values
        rebalance_dates = pd.DatetimeIndex(rebalance_dates).sort_values()
        rebalance_dates = rebalance_dates[rebalance_dates < all_dates[-1]]

        date_pos = {d: i for i, d in enumerate(all_dates)}
        long_group_idx = n_groups - 1 if factor_positive else 0
        short_group_idx = 0 if factor_positive else n_groups - 1

        # 每次调仓：在调仓日分组选股，持有到下次调仓日前一日
        for k, reb_date in enumerate(rebalance_dates):
            reb_pos = date_pos[reb_date]
            next_reb_date = rebalance_dates[k + 1] if k + 1 < len(rebalance_dates) else None
            hold_start = reb_pos
            hold_end = date_pos[next_reb_date] if next_reb_date is not None else len(all_dates) - 1
            if hold_start >= hold_end:
                continue

            f = factor.loc[reb_date]
            ind = industry.loc[reb_date]
            valid = f.notna() & ind.notna()
            if valid.sum() < threshold:
                continue

            reb_df = pd.DataFrame({
                'factor': f[valid],
                'industry': ind[valid].astype(str)
            })

            ind_group_stocks: Dict[str, Dict[int, List[str]]] = {}
            for ind_name, sub in reb_df.groupby('industry'):
                if len(sub) < n_groups:
                    continue
                labels = pd.qcut(sub['factor'], n_groups, labels=False, duplicates='drop')
                if labels.nunique() < 2:
                    continue
                group_map: Dict[int, List[str]] = {}
                for g in sorted(labels.unique()):
                    group_map[int(g)] = labels[labels == g].index.tolist()
                if group_map:
                    ind_group_stocks[ind_name] = group_map

            common_inds = set(ind_group_stocks.keys()).intersection(set(ind_weight_series.index))
            if not common_inds:
                continue

            # 对当日可用行业权重重新归一化，保证行业权重和为1
            day_weights = ind_weight_series.loc[list(common_inds)]
            day_weights = day_weights / day_weights.sum()
            period_start_date = all_dates[hold_start + 1]
            period_end_date = all_dates[hold_end]
            holding_period_ranges.append((period_start_date, period_end_date))

            # 在持有期内逐日计算分组收益，并汇总成多头与多空收益
            for pos in range(hold_start, hold_end):
                hold_date = all_dates[pos]
                next_date = all_dates[pos + 1]
                r = forward_returns.loc[hold_date]

                day_group_returns: Dict[int, float] = {}
                for g in range(n_groups):
                    ind_ret_g: Dict[str, float] = {}
                    for ind_name in day_weights.index:
                        stocks = ind_group_stocks.get(ind_name, {}).get(g, [])
                        if len(stocks) == 0:
                            continue
                        stock_ret = r.loc[stocks].dropna()
                        if len(stock_ret) == 0:
                            continue
                        ind_ret_g[ind_name] = float(stock_ret.mean())

                    if not ind_ret_g:
                        day_group_returns[g] = np.nan
                        continue
                    w_g = day_weights.loc[list(ind_ret_g.keys())]
                    w_g = w_g / w_g.sum()
                    day_group_returns[g] = float(sum(w_g[ind_name] * ind_ret_g[ind_name] for ind_name in w_g.index))

                day_long = day_group_returns.get(long_group_idx, np.nan)
                day_short = day_group_returns.get(short_group_idx, np.nan)
                if pd.isna(day_long) or pd.isna(day_short):
                    continue

                day_long_short = 0.5 * day_long - 0.5 * day_short
                for g in range(n_groups):
                    group_rets[f'G{g+1}'].append(day_group_returns.get(g, np.nan))
                long_short_rets.append(day_long_short)
                ret_dates.append(next_date)

        if len(ret_dates) == 0:
            raise ValueError("可用样本为空，无法构建组合，请检查输入矩阵对齐和缺失值。")

        ret_index = pd.DatetimeIndex(ret_dates, name='trade_date')
        strategy_returns = pd.DataFrame(index=ret_index)
        for g in range(n_groups):
            strategy_returns[f'G{g+1}'] = pd.Series(group_rets[f'G{g+1}'], index=ret_index)
        long_group_idx = n_groups - 1 if factor_positive else 0
        strategy_returns['long_only'] = strategy_returns[f'G{long_group_idx+1}']
        strategy_returns['long_short'] = pd.Series(long_short_rets, index=ret_index)

        benchmark_nav = benchmark_nav_full.reindex(strategy_returns.index).ffill()
        if benchmark_nav.isna().all():
            raise ValueError("基准指数日期与策略日期无重叠，无法对齐沪深300净值。")
        benchmark_returns = benchmark_nav.pct_change().fillna(0.0)
        strategy_returns['benchmark'] = benchmark_returns

        ic_main_series = _calc_ic_series(main_ic_lag)
        ic_mean = float(ic_main_series.mean()) if len(ic_main_series) > 0 else np.nan
        ic_std = float(ic_main_series.std()) if len(ic_main_series) > 1 else np.nan
        ic_ir = float(ic_mean / ic_std) if (not pd.isna(ic_std) and ic_std > 0) else np.nan
        ic_t_stat = float(ic_mean / (ic_std / np.sqrt(len(ic_main_series)))) if (not pd.isna(ic_std) and ic_std > 0 and len(ic_main_series) > 1) else np.nan

        decay_rows = []
        for lag in ic_lag_list:
            lag_series = _calc_ic_series(lag)
            lag_mean = float(lag_series.mean()) if len(lag_series) > 0 else np.nan
            lag_std = float(lag_series.std()) if len(lag_series) > 1 else np.nan
            lag_ir = float(lag_mean / lag_std) if (not pd.isna(lag_std) and lag_std > 0) else np.nan
            decay_rows.append({
                '预测天数': lag,
                'IC均值': lag_mean,
                'IC标准差': lag_std,
                'IC_IR': lag_ir,
                '样本数': int(len(lag_series))
            })
        ic_decay_df = pd.DataFrame(decay_rows).sort_values('预测天数').reset_index(drop=True)
        init_ic = float(ic_decay_df.loc[ic_decay_df['预测天数'] == main_ic_lag, 'IC均值'].iloc[0]) if (ic_decay_df['预测天数'] == main_ic_lag).any() else np.nan
        if pd.isna(init_ic) or init_ic == 0:
            ic_half_life_days = np.nan
        else:
            target_abs = abs(init_ic) * 0.5
            half_candidates = ic_decay_df.loc[ic_decay_df['IC均值'] < target_abs, '预测天数'].dropna()
            ic_half_life_days = float(half_candidates.min()) if len(half_candidates) > 0 else np.nan

        # 将日收益转为净值曲线，起点统一为1
        nav = (1 + strategy_returns).cumprod()
        nav.iloc[0] = 1.0

        # 计算年化收益率
        ann_returns = {}
        annual_days = 252
        for col in strategy_returns.columns:
            series = strategy_returns[col].dropna()
            if len(series) == 0:
                ann_returns[col] = np.nan
            else:
                ann_returns[col] = float((1 + series).prod() ** (annual_days / len(series)) - 1)
        benchmark_ann = ann_returns.get('benchmark', np.nan)

        rf_df = pd.read_csv(risk_free_rate_path, encoding='utf-8-sig')
        rf_date_col = next((c for c in ['日期', 'date', 'trade_date'] if c in rf_df.columns), None)
        rf_close_col = next((c for c in ['收盘', 'close', 'CLOSE'] if c in rf_df.columns), None)
        if rf_date_col is None or rf_close_col is None:
            raise ValueError("无风险利率文件缺少日期列或收盘列。")
        rf_df[rf_date_col] = pd.to_datetime(rf_df[rf_date_col], errors='coerce')
        rf_df[rf_close_col] = pd.to_numeric(
            rf_df[rf_close_col].astype(str).str.replace('%', '', regex=False),
            errors='coerce'
        )
        rf_df = rf_df.dropna(subset=[rf_date_col, rf_close_col]).sort_values(rf_date_col)
        if rf_df.empty:
            raise ValueError("无风险利率文件为空或无法解析。")
        rf_monthly_annual = rf_df.set_index(rf_date_col)[rf_close_col] / 100.0

        # 计算报告指标：超额收益、夏普、最大回撤
        metrics_records = []
        for col in strategy_returns.columns:
            series = strategy_returns[col].dropna()
            nav_col = nav[col].dropna()
            if len(series) == 0 or len(nav_col) == 0:
                metrics_records.append({
                    '组合': col,
                    '超额收益率': np.nan,
                    '年化收益率': np.nan,
                    '夏普比率': np.nan,
                    '收益t统计量': np.nan,
                    '交易胜率': np.nan,
                    '最大回撤': np.nan
                })
                continue
            rf_lookup_index = series.index.to_period('M').to_timestamp()
            rf_annual_series = rf_monthly_annual.reindex(rf_lookup_index, method='ffill')
            if rf_annual_series.isna().any():
                rf_annual_series = rf_annual_series.fillna(method='bfill')
            if rf_annual_series.isna().any():
                raise ValueError("无风险利率无法覆盖策略收益区间。")
            ann_vol = float(series.std() * np.sqrt(annual_days))
            ann_rf = float(rf_annual_series.mean())
            ann_ret = float(ann_returns[col])
            sharpe = float((ann_ret - ann_rf) / ann_vol) if ann_vol > 0 else np.nan
            series_std = float(series.std())
            t_stat = float(series.mean() / (series_std / np.sqrt(len(series)))) if (len(series) > 1 and series_std > 0) else np.nan
            rolling_max = nav_col.cummax()
            drawdown = nav_col / rolling_max - 1.0
            max_dd = float(drawdown.min())
            excess_ret = ann_ret - benchmark_ann if not pd.isna(benchmark_ann) else np.nan
            trade_returns = []
            for period_start, period_end in holding_period_ranges:
                period_series = strategy_returns.loc[period_start:period_end, col].dropna()
                if len(period_series) == 0:
                    continue
                trade_ret = float((1 + period_series).prod() - 1.0)
                trade_returns.append(trade_ret)
            trade_win_rate = float(np.mean(np.array(trade_returns) > 0)) if len(trade_returns) > 0 else np.nan
            metrics_records.append({
                '组合': col,
                '超额收益率': excess_ret,
                '年化收益率': ann_ret,
                '夏普比率': sharpe,
                '收益t统计量': t_stat,
                '交易胜率': trade_win_rate,
                '最大回撤': max_dd
            })
        metrics_df = pd.DataFrame(metrics_records)

        fig = None
        decay_fig = None
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(11, 6))
            for g in range(n_groups):
                ax.plot(nav.index, nav[f'G{g+1}'], label=f'Long G{g+1}')
            # ax.plot(nav.index, nav['long_only'], label=f'Long-Only (=Long G{long_group_idx+1})', linestyle='--')
            ax.plot(nav.index, nav['long_short'], label='Long-Short')
            ax.plot(nav.index, nav['benchmark'], label='HS300')
            ax.set_title('Factor Strategy Net Value')
            ax.set_xlabel('Date')
            ax.set_ylabel('Net Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            decay_fig, decay_ax = plt.subplots(figsize=(8, 5))
            decay_ax.plot(ic_decay_df['预测天数'], ic_decay_df['IC均值'], marker='o')
            decay_ax.axhline(0.0, color='gray', linewidth=1.0, linestyle='--')
            decay_ax.set_title('IC Decay Curve')
            decay_ax.set_xlabel('Forecast Horizon (Days)')
            decay_ax.set_ylabel('IC')
            decay_ax.grid(True, alpha=0.3)
            decay_fig.tight_layout()

        # 输出Markdown回测报告
        start_dt = nav.index.min()
        end_dt = nav.index.max()
        safe_factor_name = "".join(ch for ch in str(factor_name) if ch not in '\\/:*?"<>|').strip()
        if not safe_factor_name:
            safe_factor_name = "factor"
        out_dir = Path(result_dir) / safe_factor_name
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"{safe_factor_name}.md"
        ic_decay_plot_path = out_dir / f"{safe_factor_name}_IC衰减.png"
        if decay_fig is not None:
            decay_fig.savefig(ic_decay_plot_path, dpi=150, bbox_inches='tight')
        report_lines = []
        report_lines.append(f"# {factor_name} 分层回测报告")
        report_lines.append("")
        report_lines.append(f"- 回测区间: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}")
        report_lines.append(f"- 调仓频率: {rebalance_freq_norm}")
        report_lines.append(f"- 分组数量: {n_groups}")
        report_lines.append("")
        report_lines.append("## IC与衰减分析")
        report_lines.append("")
        ic_mean_str = "NaN" if pd.isna(ic_mean) else f"{ic_mean:.6f}"
        ic_ir_str = "NaN" if pd.isna(ic_ir) else f"{ic_ir:.6f}"
        ic_t_str = "NaN" if pd.isna(ic_t_stat) else f"{ic_t_stat:.4f}"
        half_life_str = "NaN" if pd.isna(ic_half_life_days) else f"{ic_half_life_days:.0f}"
        report_lines.append(f"- 主IC预测期: {main_ic_lag}日")
        report_lines.append(f"- IC均值: {ic_mean_str}")
        report_lines.append(f"- IR: {ic_ir_str}")
        report_lines.append(f"- IC t统计量: {ic_t_str}")
        report_lines.append(f"- IC半衰期(交易日): {half_life_str}")
        report_lines.append("")
        report_lines.append("| 预测天数 | IC均值 | IC标准差 | IC_IR | 样本数 |")
        report_lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for _, row in ic_decay_df.iterrows():
            lag_str = f"{int(row['预测天数'])}"
            lag_ic_str = "NaN" if pd.isna(row['IC均值']) else f"{row['IC均值']:.6f}"
            lag_std_str = "NaN" if pd.isna(row['IC标准差']) else f"{row['IC标准差']:.6f}"
            lag_ir_str = "NaN" if pd.isna(row['IC_IR']) else f"{row['IC_IR']:.6f}"
            n_str = f"{int(row['样本数'])}"
            report_lines.append(f"| {lag_str} | {lag_ic_str} | {lag_std_str} | {lag_ir_str} | {n_str} |")
        if decay_fig is not None:
            report_lines.append("")
            report_lines.append(f"![IC衰减曲线]({ic_decay_plot_path.name})")
            report_lines.append("")
        report_lines.append("## 组合指标")
        report_lines.append("")
        report_lines.append("| 组合 | 超额收益率 | 年化收益率 | 夏普比率 | 收益t统计量 | 交易胜率 | 最大回撤 |")
        report_lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in metrics_df.iterrows():
            excess_str = "NaN" if pd.isna(row['超额收益率']) else f"{row['超额收益率']:.4%}"
            ann_str = "NaN" if pd.isna(row['年化收益率']) else f"{row['年化收益率']:.4%}"
            sharpe_str = "NaN" if pd.isna(row['夏普比率']) else f"{row['夏普比率']:.4f}"
            t_str = "NaN" if pd.isna(row['收益t统计量']) else f"{row['收益t统计量']:.4f}"
            win_str = "NaN" if pd.isna(row['交易胜率']) else f"{row['交易胜率']:.2%}"
            mdd_str = "NaN" if pd.isna(row['最大回撤']) else f"{row['最大回撤']:.4%}"
            report_lines.append(f"| {row['组合']} | {excess_str} | {ann_str} | {sharpe_str} | {t_str} | {win_str} | {mdd_str} |")
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        return {
            'annualized_returns': pd.Series(ann_returns).rename('annualized_return'),
            'metrics': metrics_df,
            'ic_series': ic_main_series,
            'ic_mean': ic_mean,
            'ic_ir': ic_ir,
            'ic_t_stat': ic_t_stat,
            'ic_decay': ic_decay_df,
            'ic_half_life_days': ic_half_life_days,
            'daily_returns': strategy_returns,
            'net_value': nav,
            'figure': fig,
            'ic_decay_figure': decay_fig,
            'ic_decay_plot_path': str(ic_decay_plot_path) if decay_fig is not None else None,
            'report_path': str(report_path)
        }


class FactorManager:
    """
    因子管理器
    
    管理多个因子，支持因子组合和批量计算
    
    使用示例:
        fm = FactorManager()
        fm.add_factor(MarketCapFactor())
        fm.add_factor(MomentumFactor(20))
        
        # 计算所有因子
        fm.compute_all(data_manager)
        
        # 获取因子
        mcap = fm.get_factor("market_cap")
    """
    
    def __init__(self):
        self.factors: dict = {}
    
    def add_factor(self, factor: BaseFactor) -> "FactorManager":
        """添加因子"""
        self.factors[factor.name] = factor
        return self
    
    def remove_factor(self, name: str) -> "FactorManager":
        """移除因子"""
        if name in self.factors:
            del self.factors[name]
        return self
    
    def compute_all(self, dm: "DataManager") -> dict:
        """
        计算所有因子
        
        Returns:
            dict，键为因子名，值为因子矩阵
        """
        results = {}
        for name, factor in self.factors.items():
            print(f"计算因子: {name}...")
            factor.factor_data = factor.compute(dm)
            results[name] = factor.factor_data
        return results
    
    def get_factor(self, name: str) -> Optional[pd.DataFrame]:
        """获取因子数据"""
        if name in self.factors:
            return self.factors[name].factor_data
        return None
    
    def list_factors(self) -> List[str]:
        """列出所有因子名称"""
        return list(self.factors.keys())
    
    def combine_factors(self, 
                        weights: Optional[dict] = None,
                        method: str = "zscore") -> pd.DataFrame:
        """
        组合多个因子
        
        Args:
            weights: 权重字典 {因子名: 权重}，默认等权
            method: 组合前的标准化方法
        
        Returns:
            组合因子矩阵
        """
        if not self.factors:
            return pd.DataFrame()
        
        if weights is None:
            weights = {name: 1.0 / len(self.factors) for name in self.factors}
        
        combined = None
        for name, factor in self.factors.items():
            if factor.factor_data is None:
                continue
            
            # 标准化
            standardized = FactorUtils.standardize(factor.factor_data, method)
            weighted = standardized * weights.get(name, 0)
            
            if combined is None:
                combined = weighted
            else:
                combined = combined.add(weighted, fill_value=0)
        
        return combined
