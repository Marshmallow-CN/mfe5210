"""
量化回测系统 - 数据加载器
负责从Tushare获取数据并存储为Parquet格式

设计思路：
- 下载时按日期循环（受API限制），最终合并为单个parquet文件
- 支持增量更新：检测已有数据的最新日期，只拉取新数据
- 读取时直接加载整个parquet，用户自行筛选日期范围
"""
import pandas as pd
import numpy as np
import tushare as ts
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
import time
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import TUSHARE_TOKEN, DATA_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TushareDataLoader:
    """
    Tushare数据加载器
    
    数据存储结构：
        storage/
        ├── daily.parquet        # 全部日线数据
        ├── daily_basic.parquet  # 全部每日指标数据
        ├── stock_list.parquet   # 股票列表
        └── trade_cal.parquet    # 交易日历
    
    使用方法：
        loader = TushareDataLoader()
        
        # 首次下载（2010-2025全量）
        loader.download_all("20100101", "20251231")
        
        # 增量更新
        loader.update()
        
        # 读取数据
        daily = loader.load_daily()
        basic = loader.load_daily_basic()
    """
    
    # 文件路径
    DAILY_FILE = "daily.parquet"
    BASIC_FILE = "daily_basic.parquet"
    STOCK_FILE = "stock_list.parquet"
    CALENDAR_FILE = "trade_cal.parquet"
    
    def __init__(self, token: str = TUSHARE_TOKEN, data_dir: Path = DATA_DIR):
        ts.set_token(token)
        self.pro = ts.pro_api(token)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TushareDataLoader 初始化完成，数据目录: {self.data_dir}")
    
    # ==================== 文件路径 ====================
    
    @property
    def daily_path(self) -> Path:
        return self.data_dir / self.DAILY_FILE
    
    @property
    def basic_path(self) -> Path:
        return self.data_dir / self.BASIC_FILE
    
    @property
    def stock_path(self) -> Path:
        return self.data_dir / self.STOCK_FILE
    
    @property
    def calendar_path(self) -> Path:
        return self.data_dir / self.CALENDAR_FILE
    
    # ==================== 元数据获取 ====================
    
    def fetch_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表（包含上市、退市、暂停上市）
        """
        dfs = []
        for status in ['L', 'D', 'P']:  # 上市、退市、暂停
            df = self.pro.stock_basic(
                exchange='',
                list_status=status,
                fields='ts_code,symbol,name,area,industry,list_date,delist_date,market,exchange'
            )
            df['list_status'] = status
            dfs.append(df)
            time.sleep(0.1)
        
        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"获取股票列表: {len(result)} 只")
        return result
    
    def fetch_trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取交易日历
        """
        df = self.pro.trade_cal(
            exchange='SSE',
            start_date=start_date,
            end_date=end_date,
            fields='cal_date,is_open,pretrade_date'
        )
        return df.sort_values('cal_date').reset_index(drop=True)
    
    def get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        获取指定范围内的交易日列表
        """
        cal = self.fetch_trade_calendar(start_date, end_date)
        
        trade_dates = cal[cal['is_open'] == 1]['cal_date'].tolist()
        return sorted(trade_dates)
    
    # ==================== 数据下载 ====================
    
    def _fetch_daily_by_date(self, trade_date: str) -> pd.DataFrame:
        """获取某一天所有股票的日线数据"""
        df = self.pro.daily(trade_date=trade_date)
        return df if df is not None else pd.DataFrame()
    
    def _fetch_basic_by_date(self, trade_date: str) -> pd.DataFrame:
        """获取某一天所有股票的每日指标"""
        df = self.pro.daily_basic(
            trade_date=trade_date,
            fields='ts_code,trade_date,close,turnover_rate,turnover_rate_f,'
                   'volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,'
                   'total_share,float_share,free_share,total_mv,circ_mv'
        )
        return df if df is not None else pd.DataFrame()
    
    def download_daily(self, start_date: str, end_date: str, 
                       sleep_time: float = 0.15) -> pd.DataFrame:
        """
        下载日线数据
        下载的时候只下载交易日的数据
        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            sleep_time: API调用间隔（秒）
        
        Returns:
            合并后的DataFrame
        """
        trade_dates = self.get_trade_dates(start_date, end_date)
        total = len(trade_dates)
        logger.info(f"开始下载日线数据: {start_date} ~ {end_date}, 共 {total} 个交易日")
        
        dfs = []
        for i, date in enumerate(trade_dates):
            try:
                df = self._fetch_daily_by_date(date)
                if len(df) > 0:
                    dfs.append(df)
                
                if (i + 1) % 100 == 0 or i == total - 1:
                    logger.info(f"日线下载进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
                
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"下载日线数据失败 {date}: {e}")
                time.sleep(1)
        
        if not dfs:
            return pd.DataFrame()
        
        result = pd.concat(dfs, ignore_index=True)
        result['trade_date'] = pd.to_datetime(result['trade_date'], format='%Y%m%d')
        result = result.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        
        logger.info(f"日线数据下载完成: {len(result)} 条记录")
        return result
    
    def download_daily_basic(self, start_date: str, end_date: str,
                             sleep_time: float = 0.2) -> pd.DataFrame:
        """
        下载每日指标数据（市值、PE、PB等）
        
        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            sleep_time: API调用间隔（秒）
        
        Returns:
            合并后的DataFrame
        """
        trade_dates = self.get_trade_dates(start_date, end_date)
        total = len(trade_dates)
        logger.info(f"开始下载每日指标: {start_date} ~ {end_date}, 共 {total} 个交易日")
        
        dfs = []
        for i, date in enumerate(trade_dates):
            try:
                df = self._fetch_basic_by_date(date)
                if len(df) > 0:
                    dfs.append(df)
                
                if (i + 1) % 100 == 0 or i == total - 1:
                    logger.info(f"每日指标下载进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
                
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"下载每日指标失败 {date}: {e}")
                time.sleep(1)
        
        if not dfs:
            return pd.DataFrame()
        
        result = pd.concat(dfs, ignore_index=True)
        result['trade_date'] = pd.to_datetime(result['trade_date'], format='%Y%m%d')
        result = result.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        
        logger.info(f"每日指标下载完成: {len(result)} 条记录")
        return result
    
    def download_all(self, start_date: str = "20100101", end_date: str = "20251219") -> None:
        """
        下载所有数据并保存
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        """
        logger.info("=" * 60)
        logger.info(f"开始全量下载: {start_date} ~ {end_date}")
        logger.info("=" * 60)
        
        # 1. 下载并保存交易日历
        logger.info("\n[1/4] 下载交易日历...")
        cal = self.fetch_trade_calendar(start_date, end_date)
        cal.to_parquet(self.calendar_path, index=False)
        logger.info(f"交易日历已保存: {len(cal)} 条")
        
        # 2. 下载并保存股票列表
        logger.info("\n[2/4] 下载股票列表...")
        stocks = self.fetch_stock_list()
        stocks.to_parquet(self.stock_path, index=False)
        logger.info(f"股票列表已保存: {len(stocks)} 只")
        
        # 3. 下载并保存日线数据
        logger.info("\n[3/4] 下载日线数据...")
        daily = self.download_daily(start_date, end_date)
        if len(daily) > 0:
            daily.to_parquet(self.daily_path, index=False)
            logger.info(f"日线数据已保存: {self.daily_path}")
        
        # 4. 下载并保存每日指标
        logger.info("\n[4/4] 下载每日指标...")
        basic = self.download_daily_basic(start_date, end_date)
        if len(basic) > 0:
            basic.to_parquet(self.basic_path, index=False)
            logger.info(f"每日指标已保存: {self.basic_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("全部数据下载完成！")
        logger.info("=" * 60)
        self.info()
    
    # ==================== 增量更新 ====================
    
    def _get_last_date(self, file_path: Path) -> Optional[str]:
        """获取parquet文件中的最新日期"""
        if not file_path.exists():
            return None
        
        df = pd.read_parquet(file_path, columns=['trade_date'])
        if len(df) == 0:
            return None
        
        last_date = df['trade_date'].max()
        if isinstance(last_date, pd.Timestamp):
            return last_date.strftime('%Y%m%d')
        return str(last_date)
    
    def update(self, end_date: Optional[str] = None) -> None:
        """
        增量更新数据
        
        检测本地数据的最新日期，下载之后的新数据并追加
        
        Args:
            end_date: 更新到的日期，默认为今天
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        logger.info(f"开始增量更新，目标日期: {end_date}")
        
        # 更新日线数据
        last_daily = self._get_last_date(self.daily_path)
        if last_daily:
            # 从最后日期的下一天开始
            start = (pd.to_datetime(last_daily) + pd.Timedelta(days=1)).strftime('%Y%m%d')
            if start <= end_date:
                logger.info(f"日线数据: 从 {start} 更新到 {end_date}")
                new_daily = self.download_daily(start, end_date)
                if len(new_daily) > 0:
                    old_daily = pd.read_parquet(self.daily_path)
                    combined = pd.concat([old_daily, new_daily], ignore_index=True)
                    combined.to_parquet(self.daily_path, index=False)
                    logger.info(f"日线数据已更新: 新增 {len(new_daily)} 条")
            else:
                logger.info(f"日线数据已是最新 (最后日期: {last_daily})")
        else:
            logger.warning("未找到日线数据，请先执行 download_all()")
        
        # 更新每日指标
        last_basic = self._get_last_date(self.basic_path)
        if last_basic:
            start = (pd.to_datetime(last_basic) + pd.Timedelta(days=1)).strftime('%Y%m%d')
            if start <= end_date:
                logger.info(f"每日指标: 从 {start} 更新到 {end_date}")
                new_basic = self.download_daily_basic(start, end_date)
                if len(new_basic) > 0:
                    old_basic = pd.read_parquet(self.basic_path)
                    combined = pd.concat([old_basic, new_basic], ignore_index=True)
                    combined.to_parquet(self.basic_path, index=False)
                    logger.info(f"每日指标已更新: 新增 {len(new_basic)} 条")
            else:
                logger.info(f"每日指标已是最新 (最后日期: {last_basic})")
        else:
            logger.warning("未找到每日指标数据，请先执行 download_all()")
        
        # 更新股票列表
        logger.info("更新股票列表...")
        stocks = self.fetch_stock_list()
        stocks.to_parquet(self.stock_path, index=False)
        
        logger.info("增量更新完成！")
    
    # ==================== 数据读取 ====================
    
    def load_daily(self) -> pd.DataFrame:
        """
        加载全部日线数据
        
        Returns:
            DataFrame，包含字段: ts_code, trade_date, open, high, low, close, 
                                pre_close, change, pct_chg, vol, amount
        """
        if not self.daily_path.exists():
            logger.error(f"日线数据文件不存在: {self.daily_path}")
            return pd.DataFrame()
        
        df = pd.read_parquet(self.daily_path)
        logger.info(f"加载日线数据: {len(df)} 条, "
                   f"{df['ts_code'].nunique()} 只股票, "
                   f"{df['trade_date'].nunique()} 个交易日")
        return df
    
    def load_daily_basic(self) -> pd.DataFrame:
        """
        加载全部每日指标数据
        
        Returns:
            DataFrame，包含字段: ts_code, trade_date, close, turnover_rate, 
                                pe, pb, ps, total_mv, circ_mv 等
        """
        if not self.basic_path.exists():
            logger.error(f"每日指标文件不存在: {self.basic_path}")
            return pd.DataFrame()
        
        df = pd.read_parquet(self.basic_path)
        logger.info(f"加载每日指标: {len(df)} 条")
        return df
    
    def load_stock_list(self) -> pd.DataFrame:
        """加载股票列表"""
        if not self.stock_path.exists():
            return self.fetch_stock_list()
        return pd.read_parquet(self.stock_path)
    
    def load_trade_calendar(self) -> pd.DataFrame:
        """加载交易日历"""
        if not self.calendar_path.exists():
            logger.error("交易日历文件不存在")
            return pd.DataFrame()
        return pd.read_parquet(self.calendar_path)
    
    # ==================== 工具方法 ====================
    
    def info(self) -> None:
        """打印数据概览"""
        print("\n" + "=" * 50)
        print("数据概览")
        print("=" * 50)
        
        for name, path in [
            ("日线数据", self.daily_path),
            ("每日指标", self.basic_path),
            ("股票列表", self.stock_path),
            ("交易日历", self.calendar_path),
        ]:
            if path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                df = pd.read_parquet(path)
                
                if 'trade_date' in df.columns:
                    date_range = f"{df['trade_date'].min()} ~ {df['trade_date'].max()}"
                else:
                    date_range = "N/A"
                
                print(f"\n{name}: {path.name}")
                print(f"  大小: {size_mb:.1f} MB")
                print(f"  行数: {len(df):,}")
                print(f"  日期范围: {date_range}")
            else:
                print(f"\n{name}: 文件不存在")
        
        print("\n" + "=" * 50)


class DataManager:
    """
    数据管理器 - 提供给策略和回测模块的统一数据接口
    
    将长格式数据转换为宽格式矩阵 (T x N) 返回每天每只股票的xxx矩阵(如权重矩阵)，便于向量化计算
    
    使用方法：
        dm = DataManager()
        dm.load("20200101", "20231231")  # 加载指定日期范围
        
        # 获取矩阵
        price = dm.price        # 收盘价矩阵
        returns = dm.returns    # 收益率矩阵
        basic = dm.basic      # 基础指标矩阵
        mcap = dm.market_cap    # 市值矩阵
    """
    
    def __init__(self, loader: Optional[TushareDataLoader] = None):
        self.loader = loader or TushareDataLoader()
        # 矩阵缓存（宽格式）
        self._matrices: dict = {}
    
    def load(self, start_date: str, 
             end_date: str,
             data_name: str = "日线数据")  -> "DataManager":
        """
        加载指定日期范围的各类数据,并保存在dm的私有属性中
        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            data_name: 数据类型，可选 "日线数据" 或 "基础指标"等
        Returns:
            self,支持链式调用
        """
        # 转换日期格式
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        # 加载日线数据
        if data_name == "日线数据":
            daily_all = self.loader.load_daily()#加载了全部的日线数据
            # 筛选日期范围
            self._daily = daily_all[
                (daily_all['trade_date'] >= start_dt) & 
                (daily_all['trade_date'] <= end_dt)
            ].copy()
        
        if data_name == "基础指标":
            basic_all = self.loader.load_daily_basic()
            self._basic = basic_all[
                (basic_all['trade_date'] >= start_dt) & 
                (basic_all['trade_date'] <= end_dt)
            ].copy()
        
        if data_name == "股票基础信息":
            self._stock_list = self.loader.load_stock_list()

        # 清空矩阵缓存
        self._matrices = {}
        
        return self
    
    def get_matrix_from_long(self,
                    source: str,
                    field: str
                    ) -> pd.DataFrame:
        """
        获取指定字段的矩阵（带缓存）
        这里必须确保输入的数据源是长表，才能正确地进行矩阵生成;如果不是长表，先转为长表再进行矩阵生成
        """
        cache_key = f"{source}_{field}"
        #如果该字段的矩阵不存在，才进行计算
        if cache_key not in self._matrices:
            if source == '日线数据':
                data = self._daily
            elif source == '基础指标':
                data = self._basic
            else:
                logger.error(f"未知数据源: {source}")
                return pd.DataFrame()
            if data is None or data.empty or field not in data.columns:
                return pd.DataFrame()
            #pivot的意思是长转宽！！！将一行行的长数据转为矩阵的宽格式！！长表转宽表用pivot!!!
            #将这个字段的矩阵缓存如self._matrices中，key为cache_key
            self._matrices[cache_key] = data.pivot(
                index='trade_date',
                columns='ts_code',
                values=field
            )

        return self._matrices[cache_key]
    
    def get_matrix_from_unique(self,
                    source: str,
                    field: str
                    ) -> pd.DataFrame:
        '''
        从与股票代码一一对应的数据中提取指定字段的矩阵
        '''
        cache_key = f"{source}_{field}"
        #如果该字段的矩阵不存在，才进行计算
        if cache_key not in self._matrices:
            if source == '股票基础信息':
                data = self._stock_list
                if field == 'industry':
                    industry = data.set_index('ts_code')['industry']
                    industry_matrix = self.get_matrix_from_long("日线数据", "close").copy()
                    for col in industry_matrix.columns:
                        # 给【每一列的每一行】都填上 行业名称
                        industry_matrix[col] = industry.get(col, np.nan)
                    return industry_matrix

    #以下是为了方便常用的字段，直接提供属性访问,返回对应的矩阵
    @property
    def price(self) -> pd.DataFrame:
        """收盘价矩阵 (T x N)"""
        return self.get_matrix_from_long("日线数据", "close")
    
    @property
    def open(self) -> pd.DataFrame:
        """开盘价矩阵 (T x N)"""
        return self.get_matrix_from_long("日线数据", "open")
        
    @property
    def market_cap(self) -> pd.DataFrame:
        """总市值矩阵 (T x N)，单位：万元"""
        return self.get_matrix_from_long("基础指标", "total_mv")
    
    @property
    def circ_mv(self) -> pd.DataFrame:
        """流通市值矩阵 (T x N)，单位：万元"""
        return self.get_matrix_from_long("基础指标", "circ_mv")
    
    @property
    def daily_data(self) -> pd.DataFrame:
        """原始日线数据（长格式）"""
        return self._daily
    
    @property
    def basic_data(self) -> pd.DataFrame:
        """原始每日指标数据（长格式）"""
        return self._basic
    
    @property
    def industry(self) -> pd.DataFrame:
        """行业分类矩阵 (T x N)"""
        return self.get_matrix_from_unique("股票基础信息", "industry")


# ==================== 测试代码 ====================
if __name__ == "__main__":
    loader = TushareDataLoader()
    
    print("=== 测试获取交易日 ===")
    dates = loader.get_trade_dates("20251201", "20251220")
    print(f"交易日: {dates}")
    
    print("\n=== 测试获取单日数据 ===")
    df = loader._fetch_daily_by_date("20251220")
    print(f"20251220 日线数据: {len(df)} 条")
    print(df.head())
    
    print("\n=== 测试获取每日指标 ===")
    df = loader._fetch_basic_by_date("20251220")
    print(f"20251220 每日指标: {len(df)} 条")
    print(df.head())
