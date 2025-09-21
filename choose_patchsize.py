import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import PatchTSTForPrediction, PatchTSTConfig, TrainingArguments
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=== PatchTST Patch Size 对比实验脚本 ===")

# 数据加载与预处理
df = pd.read_csv("data.csv")
df = df.dropna()
df_section = df[(df.index >= 6000) & (df.index <= 12000)].reset_index(drop=True)
context_length = 200
prediction_length = 200
num_variates = 3
total_length = context_length + prediction_length

time_series_data = df_section[['x_angle_1', 'y_angle_1', 'z_angle_1']].values
scalers = {}
time_series_scaled = np.zeros_like(time_series_data)
train_split = int(len(time_series_data) * 0.8)
for i, col in enumerate(['x_angle_1', 'y_angle_1', 'z_angle_1']):
    scaler = StandardScaler()
    scaler.fit(time_series_data[:train_split, i:i+1])
    time_series_scaled[:, i:i+1] = scaler.transform(time_series_data[:, i:i+1])
    scalers[col] = scaler

time_series_np = time_series_scaled
past_values_list = []
future_values_list = []
if len(time_series_np) >= total_length:
    for i in range(len(time_series_np) - total_length + 1):
        past_slice = time_series_np[i : i + context_length]
        future_slice = time_series_np[i + context_length : i + total_length]
        past_values_list.append(past_slice.tolist())
        future_values_list.append(future_slice.tolist())
else:
    raise ValueError(f"数据不足，需要至少 {total_length} 个数据点，但只有 {len(time_series_np)} 个。")

split_point = int(len(past_values_list) * 0.8)
train_dataset = Dataset.from_dict({
    'past_values': past_values_list[:split_point],
    'future_values': future_values_list[:split_point]
})
test_dataset = Dataset.from_dict({
    'past_values': past_values_list[split_point:],
    'future_values': future_values_list[split_point:]
})

model_name = "ibm-granite/granite-timeseries-patchtst"
training_args = TrainingArguments(
    output_dir="./patchtst-finetuned-cpu",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    do_eval=True,
    eval_steps=50,
    save_steps=100,
    save_total_limit=1,
    logging_steps=10,
    seed=42,
    dataloader_pin_memory=False,
    no_cuda=True,
    dataloader_num_workers=0,
)

from transformers import Trainer

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        pred = outputs.prediction_outputs
        target = inputs['future_values']
        seq_len = pred.shape[1]
        device = pred.device
        weights = torch.linspace(1.0, 2.0, seq_len).to(device)
        mse = (pred - target) ** 2
        weighted_mse = mse * weights.unsqueeze(0).unsqueeze(-1)
        loss_main = weighted_mse.mean()
        # 物理约束项
        x_pred = pred[:, :, 0]
        y_pred = pred[:, :, 1]
        z_pred = pred[:, :, 2]
        y_phy = 0.3432 * x_pred + 1.3663 * z_pred - 127.8877
        phy_loss = torch.mean((y_pred - y_phy) ** 2)
        phy_weight = 0.00001
        loss = loss_main + phy_weight * phy_loss
        return (loss, outputs) if return_outputs else loss

# ========== Patch Size 对比实验 ==========
patch_size_list = [8, 16, 32, 64]
loss_results = []

# 新增：用于保存评价指标
eval_results = []

for patch_size in patch_size_list:
    print(f"\n=== 开始 PatchTST 训练，patch_length={patch_size} ===")
    config = PatchTSTConfig.from_pretrained(model_name)
    config.prediction_length = prediction_length
    config.context_length = context_length
    config.num_input_channels = num_variates
    config.patch_length = patch_size
    model = PatchTSTForPrediction.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    try:
        train_output = trainer.train()
        final_loss = train_output.metrics.get('train_loss', None)
        print(f"patch_length={patch_size} | 最终训练loss: {final_loss}")
        loss_results.append((patch_size, final_loss))

        # 新增：模型评价（测试集）
        # 取一批测试集样本做预测
        test_inputs = test_dataset[:20]['past_values']
        test_targets = test_dataset[:20]['future_values']
        test_inputs_tensor = torch.tensor(test_inputs, dtype=torch.float32)
        with torch.no_grad():
            model.eval()
            preds = model(past_values=test_inputs_tensor).prediction_outputs.cpu().numpy()
        targets = np.array(test_targets)
        # 计算MSE/MAE
        mse = np.mean((preds - targets) ** 2)
        mae = np.mean(np.abs(preds - targets))
        print(f"patch_length={patch_size} | 测试集MSE: {mse:.4f} | MAE: {mae:.4f}")
        eval_results.append({'patch_length': patch_size, 'train_loss': final_loss, 'test_mse': mse, 'test_mae': mae})

    except Exception as e:
        print(f"❌ patch_length={patch_size} 训练出错: {e}")
        loss_results.append((patch_size, None))

        eval_results.append({'patch_length': patch_size, 'train_loss': None, 'test_mse': None, 'test_mae': None})

print("\n=== Patch Size 与 Loss 对比结果 ===")
for p, l in loss_results:
    print(f"patch_length={p} | train_loss={l}")

# 新增：输出所有patch_size的评价指标
print("\n=== Patch Size 评价指标汇总 ===")
import pandas as pd
eval_df = pd.DataFrame(eval_results)
print(eval_df)

# 可视化 patch_size 与 MSE/MAE
try:
    plt.figure(figsize=(8,5))
    plt.plot(eval_df['patch_length'], eval_df['test_mse'], marker='o', label='测试集MSE')
    plt.plot(eval_df['patch_length'], eval_df['test_mae'], marker='s', label='测试集MAE')
    plt.xlabel('patch_length')
    plt.ylabel('误差指标')
    plt.title('Patch Size 对测试集误差影响')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"误差指标画图失败: {e}")

try:
    ps = [x[0] for x in loss_results if x[1] is not None]
    ls = [x[1] for x in loss_results if x[1] is not None]
    plt.figure(figsize=(8,5))
    plt.plot(ps, ls, marker='o')
    plt.xlabel('patch_length')
    plt.ylabel('train_loss')
    plt.title('Patch Size 对训练Loss影响')
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"画图失败: {e}")