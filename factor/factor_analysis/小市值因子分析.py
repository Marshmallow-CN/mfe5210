#### 第一步 导入因子分析范围的数据
##### 获取当前项目路径
import sys,pandas,numpy,os
from pathlib import Path
project_path = Path(__file__).resolve().parents[2]
##### 将当前项目路径添加到系统路径中
sys.path.insert(0,str(project_path))
##### 导入数据加载器
from data.loader import DataManager
#创建数据加载器实例
dm = DataManager()
dm.load('20250101','20251219','日线数据')
dm.load('20250101','20251219','基础指标')
dm.load('20250101','20251219',"股票基础信息")
##### 计算因子值
from factor.size import MarketCapFactor
from factor.base import FactorUtils, BaseFactor
#挑选要计算的因子
macp = MarketCapFactor(negative=False)
macp_factor_data = macp.compute(dm)
#### 对因子进行分析前的预处理，纯化因子
##### 中位数处理极值
##### Z-score标准化归一化
##### 市值中性化、行业中性化
factor = FactorUtils().winsorize(
    macp_factor_data,
    method = 'mad'
    )
industry_matrix = dm.get_matrix_from_unique("股票基础信息", "industry")
factor = FactorUtils().neutralize(factor, industry_matrix, market_cap=None)
factor = FactorUtils().standardize(factor, method="zscore")
#### 进行因子分析
##### 计算因子的IC、IR与t值
##### 绘制因子的IC、IR与t值的直方图
##### 计算因子分组收益
##### 绘制因子分组收益的直方图
industry = dm.industry
from factor.factor_analysis.factor_tools import get_hs300_industry_weight_from_excel
w = get_hs300_industry_weight_from_excel()
from factor.factor_analysis.factor_tools import get_hs300_close_from_excel
hs300_df = get_hs300_close_from_excel("2025-01-01", "2025-12-19")
factor_name = "小市值因子"
res = FactorUtils.calc_factor_returns(
    factor=factor,                     # TxN
    industry=industry,                 # TxN
    close=dm.price,                       # TxN
    hs300_industry_weight=w,  # Series 或 DataFrame
    hs300_index_daily=hs300_df,     # 含 close 或 pct_chg
    n_groups=5,
    factor_positive=False,
    rebalance_freq="M",
    factor_name="小市值因子",
    plot=True
)

import matplotlib.pyplot as plt

result_dir = Path(rf"d:\360MoveData\Users\Lenovo\Desktop\cuhksz\quant_backtest_claude\factor\factor_result\{factor_name}")
result_dir.mkdir(parents=True, exist_ok=True)

if res.get("figure") is not None:
    res["figure"].savefig(result_dir / "小市值因子_净值曲线.png", dpi=150, bbox_inches="tight")

if res.get("ic_decay_figure") is not None:
    res["ic_decay_figure"].savefig(result_dir / "小市值因子_IC衰减曲线.png", dpi=150, bbox_inches="tight")

plt.show()



























