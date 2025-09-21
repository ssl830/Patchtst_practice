import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import PatchTSTForPrediction, PatchTSTConfig, TrainingArguments
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=== PatchTST 优化器与学习率分组实验脚本 ===")

# 数据加载与预处理
# 数据区间与choose_patchsize.py一致
# 为加快速度，训练集400组，测试集100组

df = pd.read_csv("data.csv")
df = df.dropna()
df_section = df[(df.index >= 6000) & (df.index <= 12000)].reset_index(drop=True)
context_length = 200
prediction_length = 200
num_variates = 3
total_length = context_length + prediction_length

train_split = int(len(df_section) * 0.8)
time_series_data = df_section[['x_angle_1', 'y_angle_1', 'z_angle_1']].values
scalers = {}
time_series_scaled = np.zeros_like(time_series_data)
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

max_train = 3000
max_test = 1200
split_point = int(len(past_values_list) * 0.8)
train_pv = past_values_list[:split_point][:max_train]
train_fv = future_values_list[:split_point][:max_train]
test_pv = past_values_list[split_point:][:max_test]
test_fv = future_values_list[split_point:][:max_test]

train_dataset = Dataset.from_dict({
    'past_values': train_pv,
    'future_values': train_fv
})
test_dataset = Dataset.from_dict({
    'past_values': test_pv,
    'future_values': test_fv
})

model_name = "ibm-granite/granite-timeseries-patchtst"

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

# 优化器与学习率分组
optimizer_list = ["adamw_torch", "sgd"]
learning_rate_list = [1e-4, 5e-5, 1e-5]

results = []
for optimizer in optimizer_list:
    for lr in learning_rate_list:
        print(f"\n=== 训练: optimizer={optimizer}, learning_rate={lr} ===")
        training_args = TrainingArguments(
            output_dir=f"./patchtst-optlr-exp/{optimizer}-{lr}",
            num_train_epochs=2,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=lr,
            do_eval=True,
            eval_steps=50,
            save_steps=100,
            save_total_limit=1,
            logging_steps=10,
            seed=42,
            dataloader_pin_memory=False,
            no_cuda=True,
            dataloader_num_workers=0,
            optim=optimizer,
        )
        config = PatchTSTConfig.from_pretrained(model_name)
        config.prediction_length = prediction_length
        config.context_length = context_length
        config.num_input_channels = num_variates
        config.patch_length = 16  # patch_size已确定为16
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
            # 测试集评价
            test_inputs = test_dataset[:10]['past_values']
            test_targets = test_dataset[:10]['future_values']
            test_inputs_tensor = torch.tensor(test_inputs, dtype=torch.float32)
            with torch.no_grad():
                model.eval()
                preds = model(past_values=test_inputs_tensor).prediction_outputs.cpu().numpy()
            targets = np.array(test_targets)
            mse = np.mean((preds - targets) ** 2)
            mae = np.mean(np.abs(preds - targets))
            print(f"MSE: {mse:.4f} | MAE: {mae:.4f}")
            results.append({
                'optimizer': optimizer,
                'learning_rate': lr,
                'train_loss': final_loss,
                'test_mse': mse,
                'test_mae': mae
            })
        except Exception as e:
            print(f"❌ 训练出错: {e}")
            results.append({
                'optimizer': optimizer,
                'learning_rate': lr,
                'train_loss': None,
                'test_mse': None,
                'test_mae': None
            })

# 输出结果表格
print("\n=== 优化器与学习率分组实验结果 ===")
results_df = pd.DataFrame(results)
print(results_df)

# 可视化：不同优化器与学习率组合的MSE
try:
    plt.figure(figsize=(8,5))
    for optimizer in optimizer_list:
        subset = results_df[results_df['optimizer'] == optimizer]
        plt.plot(
            subset['learning_rate'],
            subset['test_mse'],
            marker='o',
            label=f'optimizer={optimizer}'
        )
    plt.xlabel('learning_rate')
    plt.ylabel('测试集MSE')
    plt.title('优化器与学习率组合对测试误差影响')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"可视化失败: {e}")
