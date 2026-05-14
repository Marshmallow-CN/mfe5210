
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from .base import BaseStrategy, StrategyConfig
from data.loader import DataManager
import logging

logger = logging.getLogger(__name__)

class Limit2LeaderStrategy(BaseStrategy):
    """
    2连板龙头策略
    
    策略逻辑：
    1. 选股：每日选出连板数最高的股票（至少2板）作为龙头
    2. 买入：次日开盘买入
    3. 卖出：
       - 不涨停：且（持仓>=2天 或 盈利>0）
       - 跌停：强制卖出（实盘可能卖不掉，回测这里简化处理）
    
    参考：策略文档/2连板龙头策略.txt
    """
    
    def __init__(self, 
                 n_stocks: int = 10,
                 name: str = "limit_2_leader"):
        super().__init__(name, StrategyConfig(n_stocks=n_stocks))
        self.n_stocks = n_stocks

    def generate_signal(self, dm: DataManager) -> pd.DataFrame:
        """
        生成交易信号
        """
        # 1. 获取数据矩阵
        close = dm.get_matrix('close')
        open_ = dm.get_matrix('open')
        high = dm.get_matrix('high')
        low = dm.get_matrix('low')
        pre_close = dm.get_matrix('pre_close')
        pct_chg = dm.get_matrix('pct_chg')
        vol = dm.get_matrix('vol')
        
        # 2. 预处理：识别涨停和连板数
        # 近似涨停：涨幅 > 9.5% 且 收盘价等于最高价
        # 注意：这里未区分ST(5%)和科创/创业(20%)，仅做通用近似
        is_limit_up = (pct_chg > 9.5) & (close == high)
        
        # 计算连板数 (Consecutive Limit Up)
        # 这是一个路径依赖的计算，需要循环或巧妙的向量化
        # 这里使用循环计算每日连板数矩阵
        limit_counts = pd.DataFrame(0, index=close.index, columns=close.columns)
        
        # 股票过滤列表
        valid_stocks = self._get_valid_stocks(dm)
        # 只保留有效股票
        valid_columns = [c for c in close.columns if c in valid_stocks]
        
        # 3. 逐日循环模拟
        signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        
        # 持仓状态: {stock_code: {'days_held': int, 'cost': float}}
        holdings = {}
        
        # 连板计数器 (Series)
        current_counts = pd.Series(0, index=close.columns)
        
        dates = close.index
        for i in range(len(dates)):
            date = dates[i]
            
            # --- A. 更新连板数 ---
            # 获取当日涨停情况
            today_limit = is_limit_up.loc[date]
            
            # 如果涨停，计数+1；否则清零
            # 注意：这里需要处理停牌情况吗？策略中fill_paused=True，这里简单处理
            current_counts = current_counts.where(today_limit, 0) + today_limit.astype(int)
            limit_counts.loc[date] = current_counts
            
            # --- B. 卖出逻辑 (检查现有持仓) ---
            # 策略：Run daily sell at 14:50
            # 此时已知当日价格情况
            
            stocks_to_sell = []
            
            for stock in list(holdings.keys()):
                if stock not in close.columns:
                    continue
                    
                info = holdings[stock]
                
                # 获取当日数据
                curr_price = close.loc[date, stock]
                is_limit = today_limit[stock]
                
                # 更新持仓天数 (假设从买入后第二天开始算持有1天? 策略代码: target_date = start_date + 2 days)
                # 简单处理：每次循环+1
                info['days_held'] += 1
                
                # 计算收益率
                cost = info['cost']
                ret = (curr_price / cost) - 1 if cost > 0 else 0
                
                # 卖出条件
                # 1. 不涨停
                if not is_limit:
                    # 2. 持有>=2天 或 盈利>0
                    # 策略中的days=2是自然日还是交易日？代码 get_shifted_date(..., 'T') 是交易日
                    if info['days_held'] >= 2 or ret > 0:
                        stocks_to_sell.append(stock)
                
                # 跌停强制卖出 (可选)
                # if is_limit_down: stocks_to_sell.append(stock)
            
            # 执行卖出
            for stock in stocks_to_sell:
                del holdings[stock]
            
            # --- C. 买入逻辑 (检查是否有空位) ---
            # 策略：Run daily get_stock_list at 9:01 (using previous date data)
            # Run daily buy at 09:30
            
            # 在我们的循环中，'date' 是 Day T.
            # 此时我们做的是 Day T 的决策吗？
            # 如果我们在 Day T 决定买入，通常是在 Day T 开盘买入。
            # 但我们需要 Day T-1 的数据来选股。
            # 所以：在处理 Day T 时，我们基于 Day T-1 的数据选出候选股，并在 Day T 开盘买入。
            
            # 但是，上面的 "卖出逻辑" 使用了 Day T 的数据 (14:50)。
            # 这意味着我们是在 Day T 的收盘后回顾？
            # 不，BacktestEngine 的 signal[t] 决定 t+1 的持仓。
            # 如果我们要模拟 "Day T 14:50 卖出"，意味着 Day T 结束时我们不再持有。
            # 所以 signal[t] 应该为 0。
            # 如果我们要模拟 "Day T 09:30 买入"，意味着 Day T 结束时我们持有。
            # 所以 signal[t] 应该为 1。
            
            # 选股逻辑：基于 T-1 的数据
            # 如果 i=0，没有 T-1，无法买入
            
            if i > 0:
                prev_date = dates[i-1]
                prev_counts = limit_counts.loc[prev_date]
                
                # 筛选候选股
                # 1. 连板数 >= 2
                # 2. 是当天的最高板 (M)
                # 3. 在有效股票池中
                
                # 仅考虑有效股票
                valid_prev_counts = prev_counts[prev_counts.index.isin(valid_stocks)]
                
                candidates = valid_prev_counts[valid_prev_counts >= 2]
                
                targets = []
                if not candidates.empty:
                    max_board = candidates.max()
                    # 选取最高板
                    leaders = candidates[candidates == max_board].index.tolist()
                    
                    # 因子排序 (VOL5) - 这里的实现简化为随机或按代码排序
                    # 如果需要严格复现，需要计算VOL5
                    targets = leaders
                
                # 执行买入
                open_slots = self.n_stocks - len(holdings)
                if open_slots > 0:
                    for stock in targets:
                        if open_slots <= 0:
                            break
                        
                        if stock not in holdings:
                            # 买入
                            # 记录成本：Day T 的开盘价
                            buy_price = open_.loc[date, stock]
                            if pd.isna(buy_price):
                                buy_price = close.loc[date, stock] # fallback
                                
                            holdings[stock] = {
                                'days_held': 0, # 持有第0天
                                'cost': buy_price
                            }
                            open_slots -= 1
            
            # --- D. 生成信号 ---
            # 记录当日结束时的持仓
            for stock in holdings:
                signal.loc[date, stock] = 1.0
                
        return signal

    def _get_valid_stocks(self, dm: DataManager) -> set:
        """
        获取符合过滤条件的股票池
        过滤：ST, 次新股(<50天), 科创板/北交所/创业板(策略中似乎只过滤了68,8,4)
        """
        stock_list = dm.loader.load_stock_list()
        
        # 1. 过滤科创板(688), 北交所(8xx, 4xx)
        # 策略代码: stock[0] != '4' and stock[0] != '8' and stock[:2] != '68'
        def filter_kcbj(code):
            if code.startswith('68') or code.startswith('8') or code.startswith('4'):
                return False
            return True
            
        # 2. 过滤ST
        # 简单通过名称判断
        def filter_st(name):
            return 'ST' not in name
            
        # 3. 过滤次新股 (需要动态判断，这里简化为只保留上市早于回测开始日期的)
        # 由于这里是静态过滤，我们无法精确实现"上市满50天"的动态窗口
        # 简化：只保留上市日期非空的
        
        valid_stocks = set()
        for _, row in stock_list.iterrows():
            code = row['ts_code']
            name = row['name']
            
            if not filter_kcbj(code):
                continue
            if not filter_st(name):
                continue
                
            valid_stocks.add(code)
            
        return valid_stocks

