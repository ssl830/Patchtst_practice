# 自定义数据集使用指南

## 📊 数据格式要求

### 1. 数据文件格式
您的数据文件应该是CSV格式，包含以下结构：

```csv
date,feature1,feature2,feature3,...,target
2020-01-01,1.2,3.4,5.6,...,10.5
2020-01-02,1.3,3.5,5.7,...,10.6
2020-01-03,1.4,3.6,5.8,...,10.7
...
```

**重要要求：**
- 第一列必须是 `date` 列（时间戳）
- 最后一列必须是目标变量（要预测的列）
- 中间列是特征变量
- 数据应按时间顺序排列

### 2. 时间格式
日期列支持多种格式：
- `YYYY-MM-DD` (例如: 2020-01-01)
- `YYYY-MM-DD HH:MM:SS` (例如: 2020-01-01 12:00:00)
- `MM/DD/YYYY` (例如: 01/01/2020)

### 3. 数据分割
Dataset_Custom 类会自动将数据按以下比例分割：
- 训练集：70%
- 验证集：20%
- 测试集：10%

## 🔧 配置步骤

### 步骤 1：准备数据
1. 将您的CSV文件放在 `./dataset/` 目录下
2. 确保文件格式符合上述要求

### 步骤 2：修改脚本参数
编辑 `scripts/PatchTST/custom_dataset.ps1` 中的以下参数：

```powershell
# 数据配置
$data_path_name = "your_data.csv"        # 改为您的数据文件名
$model_id_name = "CustomData"            # 改为您的模型标识
$enc_in = 7                             # 改为您的特征数量（不包括date和target列）

# 预测任务配置
$target = "your_target_column"           # 在脚本中添加目标列名
```

### 步骤 3：调整模型参数（可选）
根据您的数据特点调整以下参数：

```powershell
$seq_len = 96                           # 输入序列长度
$patch_len = 16                         # 补丁长度
$stride = 8                             # 步长
$d_model = 16                           # 模型维度
$e_layers = 3                           # 编码器层数
$n_heads = 4                            # 注意力头数
```

### 步骤 4：运行训练
在PowerShell中运行：
```powershell
cd scripts/PatchTST/
.\custom_dataset.ps1
```

## 📝 参数说明

### 核心参数
- `seq_len`: 输入序列长度（历史窗口大小）
- `pred_len`: 预测序列长度（预测窗口大小）
- `enc_in`: 输入特征数量
- `features`: 预测任务类型
  - `M`: 多变量预测多变量
  - `S`: 单变量预测单变量  
  - `MS`: 多变量预测单变量

### PatchTST特定参数
- `patch_len`: 将时间序列分割成补丁的长度
- `stride`: 补丁之间的步长
- `d_model`: Transformer模型的隐藏维度
- `n_heads`: 多头注意力的头数
- `e_layers`: 编码器层数

### 训练参数
- `train_epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `dropout`: Dropout率

## 🚀 运行示例

假设您有一个名为 `sales_data.csv` 的销售数据文件：

```csv
date,price,volume,temperature,humidity,sales
2020-01-01,100,50,25,60,1000
2020-01-02,101,52,26,61,1020
...
```

您需要修改脚本中的参数：
```powershell
$data_path_name = "sales_data.csv"
$model_id_name = "SalesForecasting"
$enc_in = 4                             # 4个特征：price,volume,temperature,humidity
```

然后运行脚本即可开始训练！

## 📊 结果查看

训练完成后，您可以在以下位置找到：
- **日志文件**: `logs/LongForecasting/PatchTST_[model_id]_[seq_len]_[pred_len].log`
- **模型检查点**: `checkpoints/` 目录下
- **预测结果**: 会在训练过程中显示验证和测试的MAE、MSE等指标

## ⚠️ 常见问题

1. **内存不足**: 减小 `batch_size` 或 `seq_len`
2. **训练缓慢**: 减小 `d_model` 或 `e_layers`
3. **精度不佳**: 增加 `train_epochs` 或调整学习率
4. **数据加载错误**: 检查CSV格式和列名是否正确