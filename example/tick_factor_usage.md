# 逐笔成交因子脚本使用说明

本文说明以下三个脚本的功能、输入输出关系、使用方法和常见问题：

- `data/tick_loader.py`
- `factor/tick.py`
- `example/tick_factor_validate.py`

---

## 1. 整体流程（从逐笔CSV到可回测因子）

这套流程分为三层：

1. **特征构建层**：`tick_loader.py`  
   从逐笔成交CSV中提取并聚合日级特征，保存为 Parquet。

2. **因子封装层**：`tick.py`  
   将日级特征转换为因子矩阵（`T x N`，日期 x 股票）。

3. **验证评估层**：`tick_factor_validate.py`  
   构建特征、加载回测区间行情、计算IC和分层收益，快速验证因子有效性。

---

## 2. `data/tick_loader.py`：逐笔特征构建器

### 2.1 作用

`TickFeatureStore` 负责把目录结构为：

- `e:\a股逐笔成交数据2025\data\YYYY-MM-DD\000001.csv`

这种逐笔文件，批量转换成统一的日级特征表，并保存到：

- `storage/tick_features/daily_features.parquet`

### 2.2 核心内容

#### (1) 文件遍历

- `_list_date_dirs(root)`：列出所有交易日目录
- `_list_code_files(date_dir, codes)`：列出某日下股票CSV，可按 `codes` 过滤

#### (2) 单文件解析 `_parse_tick_file`

对每只股票某一天的CSV：

- 读取列：`time, price, qty, side`
- 清洗逻辑：
  - `price/qty` 转数值，非法转为 NaN 后丢弃
  - 仅保留 `price > 0`、`qty > 0`
  - `side` 统一大写（B/S）
  - 组合 `date + time` 生成 `datetime`
- 派生逐笔特征：
  - `ofi = side_val * qty`（B=+1，S=-1）
  - `ret = log(price).diff()`
  - `rv_tick = ret^2`
  - `lambda_tick = |Δprice| / qty`
  - `minute = datetime.floor("T")`

#### (3) 分钟聚合 `_aggregate_minute`

按 `(code, minute)` 聚合：

- `volume = sum(qty)`
- `turnover = sum(price * qty)`
- `vwap = turnover / volume`
- `ofi = sum(ofi)`
- `rv = sum(rv_tick)`
- `price_close = last(price)`

#### (4) 日级聚合 `_aggregate_daily`

按 `code` 聚合：

- `ofi_norm = sum(ofi) / sum(volume)`（订单流不平衡归一化）
- `rv_sum = sum(rv)`（日内实现波动）
- `vwap_day = 加权分钟vwap`
- `lambda_med = 当日 lambda_tick 的中位数`

#### (5) 股票代码映射

通过 `TushareDataLoader().load_stock_list()` 将 `symbol(000001)` 映射成 `ts_code(000001.SZ)`，保证后续与日线数据字段一致。

### 2.3 公开接口

#### `build_daily_features(start_date, end_date, codes=None, overwrite=False)`

- 输入：
  - `start_date/end_date`：如 `"20250103"` ~ `"20250121"`
  - `codes`：可选，股票代码列表（如 `["000001", "000002"]`）
  - `overwrite`：是否覆盖历史特征
- 输出：
  - 返回 Parquet 路径（`Path`）
- 特性：
  - 若历史文件存在且 `overwrite=False`，会追加并按 `ts_date + ts_code` 去重

#### `load_daily_features(start_date, end_date)`

- 从特征Parquet读取指定区间数据
- 返回字段：
  - `ts_date, ts_code, ofi_norm, rv_sum, vwap_day, lambda_med`

---

## 3. `factor/tick.py`：逐笔因子封装

该文件把 `tick_loader.py` 生成的日级特征，封装为框架可用的因子类，均继承 `BaseFactor`。

### 3.1 因子类

1. `IntradayOFIFactor`
   - 读取 `ofi_norm`
   - 输出：`index=ts_date, columns=ts_code, values=ofi_norm`

2. `IntradayVolFactor`
   - 读取 `rv_sum`
   - 输出：`index=ts_date, columns=ts_code, values=rv_sum`

3. `IntradayLambdaFactor`
   - 读取 `lambda_med`
   - 输出：`index=ts_date, columns=ts_code, values=lambda_med`

### 3.2 调用方式

在策略/研究代码中：

1. 先确保特征文件已由 `build_daily_features` 生成
2. 再在 `dm` 对应日期区间调用 `factor.compute(dm)` 得到矩阵

---

## 4. `example/tick_factor_validate.py`：验证脚本

### 4.1 作用

用于一键做“从特征构建到因子评估”的快速验证：

1. 构建逐笔日级特征
2. 加载回测区间日线数据
3. 计算三个逐笔因子的 IC 统计
4. 计算分层收益（`n_groups=5`）

### 4.2 当前可配置参数

- `DATE_START` / `DATE_END`
- `CODES`：
  - 设为列表：小样本快速验证
  - 设为 `None`：全市场（耗时更长）

### 4.3 输出内容

- `tick features saved: ...daily_features.parquet`
- `IC summary`（OFI / VOL / LAM）
- `Layered returns`（三类因子分组累计收益）

---

## 5. 典型使用步骤

### 5.1 快速验证（小样本）

1. 在 `tick_factor_validate.py` 里设置日期和 `CODES`
2. 运行：

```bash
python -u example/tick_factor_validate.py
```

3. 看 IC 与分层结果是否有有效样本和合理分布

### 5.2 正式验证（全量）

1. 将 `CODES = None`
2. 跑同样脚本
3. 检查：
   - 有效IC样本数
   - 分层是否有明显单调性（如 G1~G5）

---

## 6. 常见问题

### Q1: 结果全是 NaN 或空表？

常见原因：

- 股票截面太小（如只选了 8 只），导致 IC/分层阈值过滤后没有有效样本
- `ts_code` 对齐后交集太小（逐笔特征股票与日线矩阵交集不足）

建议：

- 增大 `CODES` 或设为 `None`
- 检查 `FactorUtils.calc_ic / calc_factor_returns` 的最小样本阈值设置

### Q2: 报错 `could not convert string to float: 'Price'`？

说明CSV里存在表头或脏行。当前解析器已做：

- 数值列 `to_numeric(errors="coerce")`
- `on_bad_lines="skip"`
- 非法行自动丢弃

### Q3: 运行时报 `No module named 'tushare'`？

需要安装依赖：

```bash
pip install tushare
```

---

## 7. 指标含义速览

- `ofi_norm`：订单流净方向强度（买卖主导）
- `rv_sum`：日内实现波动（不确定性/活跃度）
- `lambda_med`：单位成交量对应价格冲击（流动性/冲击成本）

这三者分别对应微观结构中的“方向、波动、流动性”三个维度，可单独用，也可组合成多因子。
