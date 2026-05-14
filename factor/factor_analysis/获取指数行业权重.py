import tushare as ts
import pandas as pd
from datetime import datetime

def get_csi300_industry_weight(token: str, year: int = 2026, month: int = 3) -> pd.DataFrame:
    """
    获取沪深300指数指定月份的行业权重分布
    
    Args:
        token: Tushare API Token
        year: 年份，默认2026年
        month: 月份，默认3月
    
    Returns:
        pd.DataFrame: 包含以下列的行业权重汇总表
            - industry: 行业名称（申万一级行业）
            - weight_pct: 行业总权重（百分比形式）
            - constituent_count: 该行业包含的成分股数量
    """
    # 初始化Tushare Pro接口
    ts.set_token(token)
    pro = ts.pro_api()
    
    # 构造查询日期范围
    start_date = f"{year}{month:02d}01"
    # 获取当月最后一天（简单处理，实际可精确计算）
    if month == 12:
        end_date = f"{year}1231"
    else:
        next_month = datetime(year, month + 1, 1)
        last_day = (next_month - pd.Timedelta(days=1)).day
        end_date = f"{year}{month+1:02d}{last_day:02d}"
    
    # 1. 获取沪深300指数成分股权重
    print(f"正在获取{year}年{month}月沪深300指数权重数据...")
    weight_df = pro.index_weight(
        index_code='399300.SZ',
        start_date=start_date,
        end_date=end_date
    )
    
    if weight_df.empty:
        raise ValueError(f"未获取到{year}年{month}月的沪深300权重数据，请检查日期范围或API权限")
    
    print(f"成功获取{len(weight_df)}条成分股权重数据")
    
    # 2. 一次性获取股票基础信息中的行业字段，避免逐股票请求触发限频
    print("正在获取成分股行业分类信息...")
    basic_frames = []
    for status in ['L', 'D', 'P']:
        basic_frames.append(
            pro.stock_basic(
                exchange='',
                list_status=status,
                fields='ts_code,industry'
            )
        )
    industry_df = pd.concat(basic_frames, ignore_index=True).drop_duplicates(subset=['ts_code'], keep='first')
    
    # 3. 合并权重和行业信息
    merged_df = weight_df.merge(industry_df, left_on='con_code', right_on='ts_code', how='left')
    merged_df['industry'] = merged_df['industry'].fillna('未知')
    
    # 4. 按行业汇总权重
    result = merged_df.groupby('industry').agg({
        'weight': 'sum',
        'ts_code': 'count'
    }).rename(columns={
        'weight': 'weight_pct',
        'ts_code': 'constituent_count'
    }).reset_index()
    
    # 5. 权重归一化（确保总和为100）
    result['weight_pct'] = result['weight_pct'] / result['weight_pct'].sum() * 100
    result = result.sort_values('weight_pct', ascending=False).reset_index(drop=True)
    
    print(f"统计完成，共涉及{len(result)}个行业")
    return result


# 使用示例
if __name__ == "__main__":
    # 请替换为你的Tushare Token
    import sys
    from pathlib import Path
    project_path = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_path))
    
    from config.settings import TUSHARE_TOKEN
    TOKEN = TUSHARE_TOKEN
    
    try:
        industry_weight = get_csi300_industry_weight(token=TOKEN, year=2026, month=3)
        print("\n===== 2026年3月沪深300指数行业权重分布 =====")
        print(industry_weight.to_string(index=False))
        
        # 可选：保存到Excel文件
        industry_weight.to_excel(r"D:\360MoveData\Users\Lenovo\Desktop\cuhksz\quant_backtest_claude\factor\factor_excel\沪深300行业权重_2026年3月.xlsx", index=False)
        print("\n结果已保存到 '沪深300行业权重_2026年3月.xlsx'")
        
    except Exception as e:
        print(f"执行失败: {e}")
