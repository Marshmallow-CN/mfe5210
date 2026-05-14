import tushare as ts
import pandas as pd
from datetime import datetime
import os

def get_csi300_close_price(token: str, start_year: int = 2019, 
                           export_excel: bool = True, filename: str = None) -> pd.DataFrame:
    """
    获取沪深300指数从指定年份至今的收盘价序列
    
    使用tushare的index_daily接口获取指数日线行情数据。
    
    Args:
        token: Tushare API Token，需确保积分>=2000
        start_year: 起始年份，默认为2019
        export_excel: 是否导出为Excel文件，默认为True
        filename: 导出文件名，如不指定则自动生成
    
    Returns:
        pd.DataFrame: 包含以下列的收盘价数据
            - trade_date: 交易日期
            - close: 收盘点位
            - open: 开盘点位
            - high: 最高点位
            - low: 最低点位
            - pct_chg: 涨跌幅（%）
            - vol: 成交量（手）
            - amount: 成交额（千元）
    
    Raises:
        ValueError: 当API调用失败或积分不足时抛出
    """
    # 初始化Tushare Pro接口
    ts.set_token(token)
    pro = ts.pro_api()
    
    # 构造日期参数
    start_date = f"{start_year}0101"
    end_date = datetime.now().strftime("%Y%m%d")
    
    print(f"正在获取沪深300指数从{start_date}至{end_date}的收盘价数据...")
    print("注：本接口需要2000积分，单次最多返回8000行记录，建议分页获取")
    
    # 获取指数日线行情
    try:
        df = pro.index_daily(
            ts_code='399300.SZ',  # 沪深300指数代码，深圳行情代码
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        error_msg = str(e)
        if "没有权限" in error_msg or "积分" in error_msg:
            raise ValueError(
                "权限不足！index_daily接口需要2000积分才能调用。\n"
                "积分获取方式：\n"
                "1. 注册新用户即可获得100积分，完善个人信息获得20积分\n"
                "2. 分享接口文档到社交媒体可获得积分\n"
                "3. 推荐注册有效用户可获得50积分/人\n"
                "4. 具体请参考: https://tushare.pro/document/1?doc_id=13"
            )
        else:
            raise ValueError(f"数据获取失败: {e}")
    
    if df.empty:
        raise ValueError(f"未获取到{start_date}至{end_date}期间的沪深300指数数据")
    
    # 按交易日期排序（升序）
    df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
    
    # 选择主要字段
    result_df = df[['trade_date', 'close', 'open', 'high', 'low', 'pct_chg', 'vol', 'amount']].copy()
    
    # 重命名列为中文（便于阅读）
    result_df.columns = ['交易日期', '收盘点位', '开盘点位', '最高点位', '最低点位', '涨跌幅(%)', '成交量(手)', '成交额(千元)']
    
    print(f"成功获取 {len(result_df)} 条交易日数据")
    print(f"时间范围: {result_df['交易日期'].iloc[0]} 至 {result_df['交易日期'].iloc[-1]}")
    print(f"最新收盘点位: {result_df['收盘点位'].iloc[-1]:.2f}")
    
    # 导出Excel文件
    if export_excel:
        if filename is None:
            filename = f"沪深300指数收盘价_{start_date}_至_{end_date}.xlsx"
        
        # 确保目录存在
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            result_df.to_excel(writer, sheet_name='沪深300收盘价', index=False)
            
            # 添加一个汇总表
            summary = pd.DataFrame({
                '指标': ['数据起始日期', '数据结束日期', '总交易日数', '起始收盘点位', '结束收盘点位', 
                        '期间涨跌幅(%)', '期间最高点位', '期间最低点位', '数据更新时间'],
                '数值': [
                    result_df['交易日期'].iloc[0],
                    result_df['交易日期'].iloc[-1],
                    len(result_df),
                    result_df['收盘点位'].iloc[0],
                    result_df['收盘点位'].iloc[-1],
                    (result_df['收盘点位'].iloc[-1] - result_df['收盘点位'].iloc[0]) / result_df['收盘点位'].iloc[0] * 100,
                    result_df['最高点位'].max(),
                    result_df['最低点位'].min(),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ]
            })
            summary.to_excel(writer, sheet_name='数据汇总', index=False)
        
        print(f"数据已导出至: {os.path.abspath(filename)}")
    
    return result_df


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
        # 获取2019年至今的收盘价数据，自动导出Excel
        csi300_data = get_csi300_close_price(
            token=TOKEN,
            start_year=2019,
            export_excel=True,
            filename=r"D:\360MoveData\Users\Lenovo\Desktop\cuhksz\quant_backtest_claude\storage\指数数据\沪深300指数收盘价_2019至今.xlsx"
        )
        
        # 打印前10行数据预览
        print("\n===== 数据预览（前10行）=====")
        print(csi300_data.head(10).to_string())
        
        # 打印后10行数据
        print("\n===== 数据预览（后10行）=====")
        print(csi300_data.tail(10).to_string())
        
        # 可选：导出为CSV格式
        # csi300_data.to_csv("沪深300指数_2019至今_收盘价.csv", index=False, encoding='utf-8-sig')
        
    except Exception as e:
        print(f"执行失败: {e}")