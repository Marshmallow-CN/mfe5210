"""
数据下载脚本

运行方式:
    cd quant_backtest
    python examples/download_data.py

注意:
    - 2010-2025年数据量较大，预计需要2-3小时
    - Tushare有API频率限制，脚本已内置延时
    - 数据会存储在 storage/ 目录下的parquet文件中
    - 支持断点续传：如果中途中断，重新运行会从上次位置继续
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import TushareDataLoader

def main():
    # ============ 参数设置 ============
    START_DATE = "20100101"  # 回测开始日期
    END_DATE = "20251219"    # 回测结束日期
    
    print("=" * 60)
    print("量化回测系统 - 数据下载")
    print("=" * 60)
    print(f"日期范围: {START_DATE} ~ {END_DATE}")
    print(f"预计数据量: ~3600个交易日 × ~5000只股票")
    print()
    
    # ============ 初始化数据加载器 ============
    loader = TushareDataLoader()
    
    # ============ 下载数据 ============
    print("开始下载数据...")
    print("这可能需要2-3小时，请耐心等待...")
    print()
    
    loader.download_all(START_DATE, END_DATE)


if __name__ == "__main__":
    main()
