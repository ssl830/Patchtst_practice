# train_patchtst.py - PatchTST 模型训练脚本
# 此文件专门负责数据处理、模型配置和微调训练

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import PatchTSTForPrediction, PatchTSTConfig, Trainer, TrainingArguments
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

print("=== PatchTST 模型训练脚本 ===")

# 第1步：加载数据集
print("正在加载数据集...")
df = pd.read_csv("data.csv")
print("数据集加载完成！")
print("数据形状:", df.shape)

# 第2步：数据预处理
print("\n--- 数据预处理 ---")
# 数据清理
df = df.dropna()

# 选择指定区间的数据
df_section = df[(df.index >= 6000) & (df.index <= 12000)].reset_index(drop=True)
print(f"选择的数据区间形状: {df_section.shape}")

# 定义参数
context_length = 200      # 历史数据长度
prediction_length = 200   # 预测长度
num_variates = 3          # 变量数量 (x, y, z)
total_length = context_length + prediction_length

# 数据标准化 - 对每个变量分别进行标准化
print("正在对每个变量分别进行标准化...")
print("原始数据统计:")
time_series_data = df_section[['x_angle_1', 'y_angle_1', 'z_angle_1']].values
for i, col in enumerate(['x_angle_1', 'y_angle_1', 'z_angle_1']):
    data = time_series_data[:, i]
    print(f"  {col}: 范围[{data.min():.4f}, {data.max():.4f}], 标准差={data.std():.4f}")

# 创建多个标准化器，对每个变量分别标准化
scalers = {}
time_series_scaled = np.zeros_like(time_series_data)

print("\n开始分别标准化...")
for i, col in enumerate(['x_angle_1', 'y_angle_1', 'z_angle_1']):
    scaler = StandardScaler()
    # 只用训练区间fit标准化器
    train_split = int(len(time_series_data) * 0.8)
    scaler.fit(time_series_data[:train_split, i:i+1])
    # 全部数据transform（训练集和测试集都用训练集fit的标准化器）
    time_series_scaled[:, i:i+1] = scaler.transform(time_series_data[:, i:i+1])
    scalers[col] = scaler
    print(f"✅ {col} 标准化完成: 均值={scaler.mean_[0]:.4f}, 标准差={scaler.scale_[0]:.4f}")

time_series_np = time_series_scaled
print(f"✅ 分别标准化完成！最终数据形状: {time_series_np.shape}")

# 保存所有标准化器，供推理时使用
with open('scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)
print("所有标准化器已保存为 scalers.pkl")

# 检查数据是否含有无效值
if np.isnan(time_series_np).any() or np.isinf(time_series_np).any():
    raise ValueError("错误：标准化后的数据中包含 nan 或 inf 值！")
else:
    print("✅ 数据健全性检查通过。")

print(f"时间序列数据形状: {time_series_np.shape}")

# 第3步：创建滑动窗口数据集
print("\n--- 创建滑动窗口数据集 ---")
past_values_list = []
future_values_list = []

if len(time_series_np) >= total_length:
    for i in range(len(time_series_np) - total_length + 1):
        # 历史数据 (context_length, num_variates)
        past_slice = time_series_np[i : i + context_length]
        # 未来数据 (prediction_length, num_variates)
        future_slice = time_series_np[i + context_length : i + total_length]
        
        past_values_list.append(past_slice.tolist())
        future_values_list.append(future_slice.tolist())
else:
    raise ValueError(f"数据不足，需要至少 {total_length} 个数据点，但只有 {len(time_series_np)} 个。")

print(f"past_values[0] 形状: {np.array(past_values_list[0]).shape}")
print(f"future_values[0] 形状: {np.array(future_values_list[0]).shape}")

# 划分训练集和测试集
split_point = int(len(past_values_list) * 0.8)

train_dataset = Dataset.from_dict({
    'past_values': past_values_list[:split_point],
    'future_values': future_values_list[:split_point]
})
test_dataset = Dataset.from_dict({
    'past_values': past_values_list[split_point:],
    'future_values': future_values_list[split_point:]
})

print(f"训练样本数: {len(train_dataset)}")
print(f"测试样本数: {len(test_dataset)}")

# 保存测试数据集，供推理时使用
with open('test_dataset.pkl', 'wb') as f:
    pickle.dump(test_dataset, f)
print("测试数据集已保存为 test_dataset.pkl")

# 第4步：加载和配置 PatchTST 模型
print("\n--- 加载和配置 PatchTST 模型 ---")
model_name = "ibm-granite/granite-timeseries-patchtst"

# 加载配置
config = PatchTSTConfig.from_pretrained(model_name)
config.prediction_length = prediction_length
config.context_length = context_length
config.num_input_channels = num_variates
config.patch_length = 16  # 已进行过实验，16效果优于8、32、64

# 加载模型
model = PatchTSTForPrediction.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True,
)

print("模型加载完成！")

# 第5步：定义训练参数
print("\n--- 定义训练参数 ---")
training_args = TrainingArguments(
    output_dir="./patchtst-finetuned-cpu",
    num_train_epochs=30,  # 增加训练轮数
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=1e-4,  # 提高学习率并更换优化器
    optim="adamw_torch",  # 更换为AdamW优化器
    # 评估参数
    do_eval=True,
    eval_steps=50,
    # 保存参数
    save_steps=100,
    save_total_limit=1,
    logging_steps=10,
    seed=42,
    dataloader_pin_memory=False,
    no_cuda=True,  # 使用CPU
    dataloader_num_workers=0,
)

# ================== 加权损失Trainer ==================
from transformers import Trainer
import torch

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 获取模型输出
        outputs = model(**inputs)
        # 预测结果 (batch_size, prediction_length, num_variates)
        pred = outputs.prediction_outputs
        target = inputs['future_values']
        # 构造权重：线性递增，远期权重更高
        seq_len = pred.shape[1]
        device = pred.device
        weights = torch.linspace(1.0, 2.0, seq_len).to(device)  # 可调整权重区间
        # 计算加权MSE损失
        mse = (pred - target) ** 2  # (batch, seq_len, num_var)
        weighted_mse = mse * weights.unsqueeze(0).unsqueeze(-1)  # 广播到所有batch和变量
        loss_main = weighted_mse.mean()

        # ===== 物理约束损失项 =====
        # 回归方程: y = 0.3432 * x + 1.3663 * z - 127.8877
        x_pred = pred[:, :, 0]  # (batch, seq_len)
        y_pred = pred[:, :, 1]
        z_pred = pred[:, :, 2]
        y_phy = 0.3432 * x_pred + 1.3663 * z_pred - 127.8877
        phy_loss = torch.mean((y_pred - y_phy) ** 2)
        phy_weight = 0.000001 # 物理约束损失权重，可根据需要调整
        loss = loss_main + phy_weight * phy_loss
        # =========================
        return (loss, outputs) if return_outputs else loss
# ================== 加权损失Trainer END ==================

# 第6步：创建 Trainer 并开始微调
print("\n--- 开始模型微调 ---")
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("开始微调模型...")
print("这可能需要较长时间，请耐心等待...")

# 开始训练
try:
    trainer.train()
    print("✅ 训练完成！")
except Exception as e:
    print(f"❌ 训练过程中出错: {e}")
    import traceback
    traceback.print_exc()

# 第7步：保存模型
print("\n--- 保存模型 ---")
trainer.save_model("./best_patchtst_model")
print("✅ 最佳模型已保存到 ./best_patchtst_model 目录")

print("\n=== 训练完成！===")
print("现在你可以使用 predict_patchtst.py 进行预测推理。")

# ========== 训练集样本预测与可视化 ==========
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def plot_train_sample_prediction(model, train_dataset, scalers, num_samples=3, device='cpu'):
    print(f"\n--- 随机抽取训练集样本进行预测与可视化（共{num_samples}组） ---")
    indices = random.sample(range(len(train_dataset)), num_samples)
    for idx in indices:
        sample = train_dataset[idx]
        past = torch.tensor([sample['past_values']], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(past_values=past)
            pred_np = pred.prediction_outputs.squeeze(0).cpu().numpy()
        actual_np = np.array(sample['future_values'])
        # 反标准化
        feature_names = ['x_angle_1', 'y_angle_1', 'z_angle_1']
        pred_original = np.zeros_like(pred_np)
        actual_original = np.zeros_like(actual_np)
        for i, feat in enumerate(feature_names):
            pred_original[:, i:i+1] = scalers[feat].inverse_transform(pred_np[:, i:i+1])
            actual_original[:, i:i+1] = scalers[feat].inverse_transform(actual_np[:, i:i+1])
        # 画图
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        variables = ['X角度', 'Y角度', 'Z角度']
        for i in range(3):
            axes[i].plot(actual_original[:, i], label='实际值', alpha=0.8, linewidth=2)
            axes[i].plot(pred_original[:, i], label='预测值', alpha=0.8, linewidth=2, linestyle='--')
            mse = mean_squared_error(actual_original[:, i], pred_original[:, i])
            mae = mean_absolute_error(actual_original[:, i], pred_original[:, i])
            axes[i].set_title(f'{variables[i]} - 训练集样本预测 (MSE: {mse:.4f}, MAE: {mae:.4f})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylabel('角度值')
        axes[-1].set_xlabel('时间步')
        plt.suptitle(f'训练集样本预测结果（样本ID: {idx}）', fontsize=16)
        plt.tight_layout()
        plt.show()

# 训练结束后自动可视化
plot_train_sample_prediction(model, train_dataset, scalers, num_samples=3, device='cpu')
