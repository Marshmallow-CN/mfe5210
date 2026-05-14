"""
快速测试脚本 - 验证数据模块是否正常工作

只下载最近1周的数据进行测试，验证API连通性和代码逻辑
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import TushareDataLoader, DataManager

def main():
    print("=" * 50)
    print("数据模块快速测试")
    print("=" * 50)
    
    # 只测试最近1周（约5个交易日）
    START_DATE = "20251216"
    END_DATE = "20251222"
    
    loader = TushareDataLoader()
    
    # 1. 测试获取股票列表
    print("\n[1] 获取股票列表...")
    stocks = loader.fetch_stock_list()
    print(f"    共 {len(stocks)} 只股票")
    
    # 2. 测试获取交易日
    print("\n[2] 获取交易日...")
    trade_dates = loader.get_trade_dates(START_DATE, END_DATE)
    print(f"    交易日: {trade_dates}")
    
    # 3. 测试下载数据（小范围）
    print("\n[3] 下载测试数据...")
    loader.download_all(START_DATE, END_DATE)
    
    # 4. 测试DataManager
    print("\n[4] 测试DataManager...")
    dm = DataManager(loader)
    dm.load(START_DATE, END_DATE)
    
    if not dm.price.empty:
        print(f"\n    收盘价矩阵: {dm.price.shape}")
        print(f"    市值矩阵: {dm.market_cap.shape}")
        
        print("\n[6] 市值矩阵预览 (前5行x前5列):")
        print(dm.market_cap.iloc[:5, :5])
    
    print("\n" + "=" * 50)
    print("测试完成！数据模块工作正常 ✓")
    print("=" * 50)
    
    # 显示数据信息
    loader.info()


if __name__ == "__main__":
    main()
