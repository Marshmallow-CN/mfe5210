"""
数据增量更新脚本

运行方式:
    cd quant_backtest
    python examples/update_data.py

功能:
    检测本地数据的最新日期，下载之后的新数据并追加到parquet文件
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import TushareDataLoader

def main():
    print("=" * 50)
    print("量化回测系统 - 数据增量更新")
    print("=" * 50)
    
    loader = TushareDataLoader()
    
    # 显示当前数据状态
    print("\n当前数据状态:")
    loader.info()
    
    # 执行增量更新
    print("\n开始增量更新...")
    loader.update()
    
    # 显示更新后状态
    print("\n更新后数据状态:")
    loader.info()


if __name__ == "__main__":
    main()
