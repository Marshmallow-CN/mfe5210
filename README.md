# 量化回测系统 v0.3

一个完整的A股量化回测框架，支持因子计算、策略开发和回测分析。

## 📁 项目结构

```
quant_backtest/
├── config/                 # 配置模块
│   ├── __init__.py
│   └── settings.py        # 全局配置（Tushare Token、回测参数等）
│
├── data/                   # 数据模块
│   ├── __init__.py
│   └── loader.py          # Tushare数据加载器 + DataManager
│
├── factor/                 # 因子模块
│   ├── __init__.py
│   ├── base.py            # 因子基类、工具类、因子管理器
│   ├── size.py            # 市值因子（总市值、流通市值、市值排名等）
│   ├── value.py           # 价值因子（PE、PB、PS、股息率等）
│   └── momentum.py        # 动量因子（动量、反转、RSI等）
│
├── strategy/               # 策略模块
│   ├── __init__.py
│   ├── base.py            # 策略基类、策略配置
│   └── small_cap.py       # 小市值策略（6种变体）
│
├── backtest/               # 回测模块 ⭐ 新增
│   ├── __init__.py
│   ├── engine.py          # 向量化回测引擎
│   ├── metrics.py         # 绩效指标计算
│   └── report.py          # 回测报告生成
│
├── example/                # 示例脚本
│   ├── demo_small_cap.py  # 因子和策略演示
│   └── demo_backtest.py   # 完整回测演示 ⭐ 新增
│
└── storage/                # 数据存储目录
```

## 🚀 快速开始

### 1. 使用模拟数据测试

```bash
cd quant_backtest

# 因子和策略演示
python example/demo_small_cap.py

# 完整回测演示
python example/demo_backtest.py
```

### 2. 使用真实数据

```python
from data.loader import TushareDataLoader, DataManager
from strategy.small_cap import SmallCapStrategy
from backtest import BacktestEngine, BacktestConfig, BacktestReport

# 下载数据（首次运行）
loader = TushareDataLoader()
loader.download_all("20200101", "20251220")

# 加载数据
dm = DataManager(loader)
dm.load("20200101", "20251220")

# 运行策略
strategy = SmallCapStrategy(n_stocks=30)
weights = strategy.run(dm)

# 执行回测
config = BacktestConfig(
    commission_rate=0.0003,
    slippage=0.001,
    stamp_duty=0.001,
)
engine = BacktestEngine(config)
result = engine.run(weights, dm)

# 生成报告
report = BacktestReport(result, "小市值策略")
report.print_summary()
```

## MFE5210 作业工作流

本仓库已经补充了面向课程作业的日频横截面 alpha 开发流程，默认做法是：

1. 加载 A 股日线、基础指标和股票基础信息
2. 计算一组候选日频 alpha 因子
3. 对因子做去极值、行业/市值中性化和标准化
4. 构建不含交易成本的 long-short 组合
5. 统计每个因子的 Sharpe、IC、最大回撤
6. 按相关性阈值筛出低相关因子集合
7. 自动输出结果汇总和 `README_assignment.md`

### 一键运行作业脚本

```bash
python example/run_mfe5210_assignment.py
```

### 主要新增文件

- `factor/daily_alpha.py`: 作业用日频 alpha 因子池
- `factor/factor_analysis/run_alpha_suite.py`: 批量评估、低相关筛选、README 生成
- `example/run_mfe5210_assignment.py`: 一键入口脚本

### 输出目录

运行后会在以下目录生成作业结果：

```text
factor/factor_result/mfe5210_alpha_suite/
├── alpha_summary.csv
├── alpha_long_short_returns.csv
├── selected_alpha_summary.csv
├── selected_alpha_correlation.csv
└── README_assignment.md
```

## 📊 因子模块

### 市值因子 (`factor/size.py`)

| 因子类 | 说明 |
|--------|------|
| `MarketCapFactor` | 总市值因子（取负对数） |
| `CircMarketCapFactor` | 流通市值因子 |
| `MarketCapRankFactor` | 市值排名因子 |
| `SizeChangeFactor` | 市值变化因子 |
| `SmallCapScoreFactor` | 小市值综合评分 |

### 价值因子 (`factor/value.py`)

| 因子类 | 说明 |
|--------|------|
| `PEFactor` | 市盈率因子（EP） |
| `PBFactor` | 市净率因子（BP） |
| `PSFactor` | 市销率因子（SP） |
| `DividendYieldFactor` | 股息率因子 |
| `ValueCompositeFactor` | 价值综合因子 |
| `SmallValueFactor` | 小市值+价值组合因子 |

### 动量因子 (`factor/momentum.py`)

| 因子类 | 说明 |
|--------|------|
| `MomentumFactor` | 动量因子（N日收益率） |
| `ReversalFactor` | 反转因子（负动量） |
| `RelativeStrengthFactor` | 相对强度因子 |
| `VolatilityAdjustedMomentumFactor` | 波动率调整动量 |
| `MomentumChangeFactor` | 动量加速度因子 |
| `MaxDrawdownFactor` | 最大回撤因子 |

### 因子工具

```python
from factor.base import FactorUtils

# 去极值
factor = FactorUtils.winsorize(factor, method="mad", n=3)

# 标准化
factor = FactorUtils.standardize(factor, method="zscore")

# 中性化（需要行业数据）
factor = FactorUtils.neutralize(factor, industry, market_cap)

# 计算IC
ic = FactorUtils.calc_ic(factor, returns, method="spearman")
```

## 🎯 策略模块

### 小市值策略 (`strategy/small_cap.py`)

| 策略类 | 说明 |
|--------|------|
| `SmallCapStrategy` | 纯小市值策略 |
| `SmallCapValueStrategy` | 小市值+低估值 |
| `SmallCapMomentumStrategy` | 小市值+动量过滤 |
| `SmallCapReversalStrategy` | 小市值+短期反转 |
| `MicroCapStrategy` | 微盘股策略（10亿以下） |
| `SmallCapRotationStrategy` | 小市值轮动策略 |

### 策略配置

```python
from strategy.base import StrategyConfig

config = StrategyConfig(
    n_stocks=30,              # 持仓股票数量
    weight_method="equal",     # 权重方式：equal/market_cap/factor
    rebalance_freq="weekly",   # 调仓频率：daily/weekly/monthly
    max_market_cap=5000000,    # 最大市值（万元）
    min_market_cap=None,       # 最小市值（万元）
)
```

### 使用工厂函数

```python
from strategy.small_cap import create_small_cap_strategy

# 创建不同变体
strategy = create_small_cap_strategy('pure', n_stocks=30)
strategy = create_small_cap_strategy('value', n_stocks=30)
strategy = create_small_cap_strategy('momentum', n_stocks=30)
strategy = create_small_cap_strategy('micro', n_stocks=20)
```

## 📉 回测模块 (`backtest/`)

### 回测引擎 (`engine.py`)

```python
from backtest import BacktestEngine, BacktestConfig

# 配置回测参数
config = BacktestConfig(
    commission_rate=0.0003,    # 佣金费率
    slippage=0.001,            # 滑点
    stamp_duty=0.001,          # 印花税
    trade_delay=1,             # 交易延迟（T+1）
    apply_limit_rules=True,    # 应用涨跌停规则
    apply_suspend_rules=True,  # 应用停牌规则
    initial_capital=1000000,   # 初始资金
)

# 执行回测
engine = BacktestEngine(config)
result = engine.run(weights, dm, benchmark_returns)
```

### 绩效指标 (`metrics.py`)

| 指标函数 | 说明 |
|----------|------|
| `calc_sharpe_ratio` | 夏普比率 |
| `calc_calmar_ratio` | 卡尔玛比率 |
| `calc_sortino_ratio` | 索提诺比率 |
| `calc_max_drawdown` | 最大回撤 |
| `calc_alpha_beta` | Alpha和Beta |
| `calc_information_ratio` | 信息比率 |
| `calc_all_metrics` | 计算所有指标 |

```python
from backtest import calc_all_metrics

metrics = calc_all_metrics(
    returns=result.portfolio_returns,
    benchmark_returns=benchmark_returns,
    turnover=result.turnover,
)
print(metrics)
```

### 回测报告 (`report.py`)

```python
from backtest import BacktestReport, compare_strategies

# 单策略报告
report = BacktestReport(result, "策略名称")
report.print_summary()           # 打印摘要
report.get_monthly_returns()     # 月度收益表
report.get_yearly_returns()      # 年度收益
report.get_drawdown_analysis()   # 回撤分析
report.save_report("report.txt") # 保存报告

# 多策略对比
results = {'策略A': result_a, '策略B': result_b}
compare_strategies(results)
```

## 📈 示例输出

```
================================================================================
策略对比
================================================================================
          累计收益    年化收益     波动率    最大回撤   夏普比率  卡尔玛比率      胜率     换手率
纯小市值    15.44%   7.50%  13.79%  11.10%   0.33   0.68  52.40%   1.79%
小市值+价值  -0.55%  -0.28%  14.32%  16.38%  -0.23  -0.02  49.20%  12.51%
小市值+动量  -3.04%  -1.55%  13.88%  22.51%  -0.33  -0.07  49.60%   5.76%
小市值+反转   2.96%   1.48%  14.15%  14.78%  -0.11   0.10  51.00%  14.07%
================================================================================
```

## 🔮 后续开发计划

- [x] 数据模块 (data)
- [x] 因子模块 (factor)
- [x] 策略模块 (strategy)
- [x] 回测模块 (backtest)
- [ ] 更多因子（质量、技术面、情绪等）
- [ ] 更多策略（多因子、行业轮动等）
- [ ] 可视化报告（HTML/PDF）
- [ ] 实盘对接

## 📝 注意事项

1. 本系统仅供学习研究使用，不构成投资建议
2. 历史回测结果不代表未来表现
3. 小市值策略近年来效果有所减弱
4. 请确保Tushare Token有效且有足够积分

## 📄 依赖

```
pandas>=1.3.0
numpy>=1.20.0
tushare>=1.2.80
scipy>=1.7.0
scikit-learn>=0.24.0  # 可选，用于因子中性化
```
