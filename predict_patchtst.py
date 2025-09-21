# predict_patchtst.py - PatchTST 模型预测推理脚本
# 此文件专门负责加载训练好的模型并进行预测推理

import pandas as pd
import numpy as np
import torch
from transformers import PatchTSTForPrediction
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

print("=== PatchTST 模型预测推理脚本 ===")

# 第1步：检查必要文件是否存在
print("正在检查必要文件...")
required_files = [
    "./best_patchtst_model",
    "scaler.pkl", 
    "test_dataset.pkl"
]

for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到必要文件: {file_path}。请先运行 train_patchtst.py 进行模型训练。")

print("✅ 所有必要文件检查通过！")

# 第2步：加载训练好的模型
print("\n--- 加载训练好的模型 ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = PatchTSTForPrediction.from_pretrained("./best_patchtst_model")
model.to(device)
model.eval()
print("✅ 模型加载完成！")

# 第3步：加载标准化器和测试数据
print("\n--- 加载标准化器和测试数据 ---")
# 加载所有标准化器
try:
    # 首先尝试加载新的多标准化器格式
    with open('scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    print("✅ 多变量标准化器加载完成！")
    use_multi_scalers = True
except FileNotFoundError:
    # 如果找不到新格式，尝试加载旧的单一标准化器
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ 单一标准化器加载完成！")
        use_multi_scalers = False
    except FileNotFoundError:
        raise FileNotFoundError("找不到标准化器文件！请先运行 train_patchtst.py")

# 加载测试数据集
with open('test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
print(f"✅ 测试数据集加载完成！包含 {len(test_dataset)} 个测试样本。")

# 第4步：进行预测
print("\n--- 进行预测推理 ---")

def inverse_transform_data(data, scalers_dict=None, single_scaler=None):
    """
    使用相应的标准化器进行反标准化
    data: 形状为 (n_samples, n_features) 的数组
    scalers_dict: 包含每个特征标准化器的字典
    single_scaler: 单一标准化器（兼容旧版本）
    """
    if scalers_dict is not None:
        # 使用多标准化器进行反标准化
        result = np.zeros_like(data)
        feature_names = ['x_angle_1', 'y_angle_1', 'z_angle_1']
        
        for i, feature in enumerate(feature_names):
            result[:, i:i+1] = scalers_dict[feature].inverse_transform(data[:, i:i+1])
        return result
    elif single_scaler is not None:
        # 使用单一标准化器进行反标准化（兼容旧版本）
        return single_scaler.inverse_transform(data)
    else:
        raise ValueError("必须提供 scalers_dict 或 single_scaler 之一")

def predict_single_sample(model, test_sample_dict, device):
    """对单个样本进行预测"""
    # 准备输入数据
    past_values = torch.tensor([test_sample_dict['past_values']], dtype=torch.float32).to(device)
    
    # 进行预测
    with torch.no_grad():
        prediction = model(past_values=past_values)
        predicted_values = prediction.prediction_outputs
    
    return predicted_values

def predict_multiple_samples(model, test_dataset, device, num_samples=5):
    """对多个样本进行预测"""
    predictions = []
    actual_values = []
    
    print(f"正在对 {num_samples} 个样本进行预测...")
    
    for i in range(min(num_samples, len(test_dataset))):
        test_sample = test_dataset[i]
        
        # 预测
        predicted = predict_single_sample(model, test_sample, device)
        predicted_np = predicted.squeeze(0).cpu().numpy()  # 移除batch维度并转为numpy
        
        # 实际值
        actual_np = np.array(test_sample['future_values'])
        
        predictions.append(predicted_np)
        actual_values.append(actual_np)
        
        if (i + 1) % 1 == 0:
            print(f"  已完成 {i + 1}/{num_samples} 个样本的预测")
    
    return predictions, actual_values

def save_prediction_to_csv(actual_values, predicted_values, filename="prediction_results.csv"):
    """
    将预测结果保存到CSV文件
    
    参数:
    - actual_values: 真实值 numpy数组，形状 (200, 3)
    - predicted_values: 预测值 numpy数组，形状 (200, 3)
    - filename: 输出文件名
    """
    print(f"\n正在保存预测结果到 {filename}...")
    
    # 创建时间步索引
    time_steps = np.arange(len(actual_values))
    
    # 构建DataFrame
    result_df = pd.DataFrame({
        'time_step': time_steps,
        'x_actual': actual_values[:, 0],
        'x_predicted': predicted_values[:, 0],
        'y_actual': actual_values[:, 1],
        'y_predicted': predicted_values[:, 1],
        'z_actual': actual_values[:, 2],
        'z_predicted': predicted_values[:, 2],
    })
    
    # 计算每个时间步的误差
    result_df['x_error'] = result_df['x_predicted'] - result_df['x_actual']
    result_df['y_error'] = result_df['y_predicted'] - result_df['y_actual']
    result_df['z_error'] = result_df['z_predicted'] - result_df['z_actual']
    
    # 计算绝对误差
    result_df['x_abs_error'] = np.abs(result_df['x_error'])
    result_df['y_abs_error'] = np.abs(result_df['y_error'])
    result_df['z_abs_error'] = np.abs(result_df['z_error'])
    
    # 保存到CSV
    result_df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"✅ 预测结果已保存到 {filename}")
    print(f"文件包含 {len(result_df)} 行数据，列信息:")
    print("  - time_step: 时间步")
    print("  - x_actual, y_actual, z_actual: 三轴真实值")
    print("  - x_predicted, y_predicted, z_predicted: 三轴预测值")
    print("  - x_error, y_error, z_error: 预测误差 (预测值-真实值)")
    print("  - x_abs_error, y_abs_error, z_abs_error: 绝对误差")
    
    return result_df

def save_batch_predictions_to_csv(predictions_batch, actuals_batch, scalers_dict=None, single_scaler=None, filename="batch_prediction_results.csv"):
    """
    将批量预测结果保存到CSV文件
    
    参数:
    - predictions_batch: 批量预测结果列表
    - actuals_batch: 批量真实值列表
    - scalers_dict: 多变量标准化器字典，用于反标准化
    - single_scaler: 单一标准化器（兼容旧版本）
    - filename: 输出文件名
    """
    print(f"\n正在保存批量预测结果到 {filename}...")
    
    all_results = []
    
    for sample_idx, (pred, actual) in enumerate(zip(predictions_batch, actuals_batch)):
        # 反标准化
        pred_original = inverse_transform_data(pred, scalers_dict, single_scaler)
        actual_original = inverse_transform_data(actual, scalers_dict, single_scaler)
        
        # 为每个样本创建数据
        for time_step in range(len(pred_original)):
            row = {
                'sample_id': sample_idx,
                'time_step': time_step,
                'x_actual': actual_original[time_step, 0],
                'x_predicted': pred_original[time_step, 0],
                'y_actual': actual_original[time_step, 1],
                'y_predicted': pred_original[time_step, 1],
                'z_actual': actual_original[time_step, 2],
                'z_predicted': pred_original[time_step, 2],
            }
            
            # 计算误差
            row['x_error'] = row['x_predicted'] - row['x_actual']
            row['y_error'] = row['y_predicted'] - row['y_actual'] 
            row['z_error'] = row['z_predicted'] - row['z_actual']
            row['x_abs_error'] = abs(row['x_error'])
            row['y_abs_error'] = abs(row['y_error'])
            row['z_abs_error'] = abs(row['z_error'])
            
            all_results.append(row)
    
    # 创建DataFrame并保存
    batch_df = pd.DataFrame(all_results)
    batch_df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"✅ 批量预测结果已保存到 {filename}")
    print(f"文件包含 {len(batch_df)} 行数据，覆盖 {len(predictions_batch)} 个样本")
    
    return batch_df

# 对第一个测试样本进行详细预测
print("对第一个测试样本进行预测...")
test_sample_0 = test_dataset[1]
predicted_values = predict_single_sample(model, test_sample_0, device)

print(f"预测形状: {predicted_values.shape}")

# 转换为 numpy 进行后处理
predicted_values_np = predicted_values.squeeze(0).cpu().numpy()  # (200, 3)
actual_values_np = np.array(test_sample_0['future_values'])      # (200, 3)

# 第5步：反标准化（还原到原始数值范围）
print("\n--- 反标准化处理 ---")
# 将预测值和真实值都转换回原始尺度
if use_multi_scalers:
    predicted_values_original = inverse_transform_data(predicted_values_np, scalers_dict=scalers)
    actual_values_original = inverse_transform_data(actual_values_np, scalers_dict=scalers)
    print("✅ 使用多变量标准化器进行反标准化完成！")
else:
    predicted_values_original = inverse_transform_data(predicted_values_np, single_scaler=scaler)
    actual_values_original = inverse_transform_data(actual_values_np, single_scaler=scaler)
    print("✅ 使用单一标准化器进行反标准化完成！")

# 第6步：计算评估指标
print("\n--- 计算评估指标 ---")

mse_scores = []
mae_scores = []
variables = ['X角度', 'Y角度', 'Z角度']

for i in range(3):
    mse = mean_squared_error(actual_values_original[:, i], predicted_values_original[:, i])
    mae = mean_absolute_error(actual_values_original[:, i], predicted_values_original[:, i])
    
    mse_scores.append(mse)
    mae_scores.append(mae)
    
    print(f"{variables[i]} - MSE: {mse:.4f}, MAE: {mae:.4f}")

avg_mse = np.mean(mse_scores)
avg_mae = np.mean(mae_scores)
print(f"\n平均 MSE: {avg_mse:.4f}")
print(f"平均 MAE: {avg_mae:.4f}")

# 第7步：可视化预测结果
print("\n--- 可视化预测结果 ---")

def plot_input_and_prediction(input_seq, true_future, pred_future, title="滑动窗口预测对比"):
    """
    可视化：先画输入的200条数据，再画后面200条的预测值和真实值
    input_seq: (200, 3) 输入序列
    true_future: (200, 3) 真实未来
    pred_future: (200, 3) 预测未来
    """
    # 此函数不再单独画图，改为返回数据，统一批量可视化
    return input_seq, true_future, pred_future, title


def plot_full_dataset_predictions_nonoverlap(model, test_dataset, device, scalers_dict=None, single_scaler=None, use_multi_scalers=True):
    """
    每隔200步作为一个输入窗口，进行一次不重叠预测，拼接全局预测曲线
    """
    print("\n--- 绘制全局不重叠窗口预测曲线 ---")
    step = 200
    num_samples = len(test_dataset)
    indices = list(range(0, num_samples, step))
    print(f"将进行 {len(indices)} 次不重叠预测...")

    all_predictions = []
    all_actuals = []

    for idx in indices:
        test_sample = test_dataset[idx]
        pred = predict_single_sample(model, test_sample, device)
        pred_np = pred.squeeze(0).cpu().numpy()
        actual_np = np.array(test_sample['future_values'])
        # 反标准化
        if use_multi_scalers:
            pred_original = inverse_transform_data(pred_np, scalers_dict=scalers_dict)
            actual_original = inverse_transform_data(actual_np, scalers_dict=scalers_dict)
        else:
            pred_original = inverse_transform_data(pred_np, single_scaler=single_scaler)
            actual_original = inverse_transform_data(actual_np, single_scaler=single_scaler)
        all_predictions.append(pred_original)
        all_actuals.append(actual_original)
        print(f"  已完成窗口 {idx} -> {idx+step}")

    # 拼接所有预测和真实值（不重叠）
    full_pred = np.concatenate(all_predictions, axis=0)
    full_actual = np.concatenate(all_actuals, axis=0)
    print(f"全局预测序列长度: {len(full_pred)} (不重叠)")

    # 绘制
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    variables = ['X角度', 'Y角度', 'Z角度']
    for i in range(3):
        axes[i].plot(full_actual[:, i], label='实际值', alpha=0.8, linewidth=1.5, color='blue')
        axes[i].plot(full_pred[:, i], label='预测值', alpha=0.8, linewidth=1.5, linestyle='--', color='red')
        mse = mean_squared_error(full_actual[:, i], full_pred[:, i])
        mae = mean_absolute_error(full_actual[:, i], full_pred[:, i])
        axes[i].set_title(f'{variables[i]} - 全局不重叠预测 (MSE: {mse:.4f}, MAE: {mae:.4f})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel('角度值')
    axes[-1].set_xlabel('时间步 (测试集起点)')
    plt.suptitle('全局不重叠窗口预测结果', fontsize=16)
    plt.tight_layout()
    plt.show()
    # 打印统计
    print(f"\n=== 全局不重叠预测统计 ===")
    for i in range(3):
        mse = mean_squared_error(full_actual[:, i], full_pred[:, i])
        mae = mean_absolute_error(full_actual[:, i], full_pred[:, i])
        print(f"{variables[i]} - MSE: {mse:.4f}, MAE: {mae:.4f}")
    overall_mse = np.mean([mean_squared_error(full_actual[:, i], full_pred[:, i]) for i in range(3)])
    overall_mae = np.mean([mean_absolute_error(full_actual[:, i], full_pred[:, i]) for i in range(3)])
    print(f"\n总体 - MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}")
    return full_pred, full_actual

# 绘制单个样本的预测结果
single_plot_data = [(
    actual_values_np,
    actual_values_original,
    predicted_values_original,
    "单个样本滑动窗口预测对比"
)]


# 绘制全局不重叠窗口预测曲线
print("\n" + "="*60)
print("随机抽取3个滑动窗口进行预测可视化...")
print("="*60)
# 获取原始验证集数据（假设test_dataset每个样本都含有past_values和future_values）
total_samples = len(test_dataset)
window_size = 200
future_size = 200
max_start = total_samples - 1
np.random.seed(42)  # 保证可复现
random_indices = np.random.choice(range(max_start), size=3, replace=False)
multi_plot_data = []
for idx, sample_idx in enumerate(random_indices):
    test_sample = test_dataset[sample_idx]
    input_seq = np.array(test_sample['past_values'])
    true_future = np.array(test_sample['future_values'])
    pred = predict_single_sample(model, test_sample, device)
    pred_np = pred.squeeze(0).cpu().numpy()
    if use_multi_scalers:
        input_seq_original = inverse_transform_data(input_seq, scalers_dict=scalers)
        true_future_original = inverse_transform_data(true_future, scalers_dict=scalers)
        pred_future_original = inverse_transform_data(pred_np, scalers_dict=scalers)
    else:
        input_seq_original = inverse_transform_data(input_seq, single_scaler=scaler)
        true_future_original = inverse_transform_data(true_future, single_scaler=scaler)
        pred_future_original = inverse_transform_data(pred_np, single_scaler=scaler)
    multi_plot_data.append((input_seq_original, true_future_original, pred_future_original, f"随机窗口{idx+1} (样本索引: {sample_idx})"))

# 一次性画出三张图（每张图三轴）
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
# 新增：收集所有窗口的后200步真实值和预测值
csv_rows = []
for row_idx, (input_seq, true_future, pred_future, title) in enumerate(multi_plot_data):
    for i in range(3):
        axes[row_idx, i].plot(np.arange(200), input_seq[:, i], label='输入(历史200)', color='blue', linewidth=2)
        axes[row_idx, i].plot(np.arange(200, 400), true_future[:, i], label='真实(未来200)', color='green', linewidth=2)
        axes[row_idx, i].plot(np.arange(200, 400), pred_future[:, i], label='预测(未来200)', color='red', linestyle='--', linewidth=2)
        axes[row_idx, i].set_title(f'{variables[i]} - {title}')
        axes[row_idx, i].legend()
        axes[row_idx, i].grid(True, alpha=0.3)
        axes[row_idx, i].set_ylabel('角度值')
    axes[row_idx, -1].set_xlabel('时间步')
    # 收集后200步真实值和预测值
    for t in range(200):
        csv_rows.append({
            'group': row_idx+1,
            'time_step': t,
            'x_actual': true_future[t, 0],
            'x_predicted': pred_future[t, 0],
            'y_actual': true_future[t, 1],
            'y_predicted': pred_future[t, 1],
            'z_actual': true_future[t, 2],
            'z_predicted': pred_future[t, 2],
        })
plt.tight_layout()
plt.show()
# 保存到csv
csv_df = pd.DataFrame(csv_rows)
csv_df.to_csv('three_windows_prediction.csv', index=False, encoding='utf-8-sig')
print('✅ 三组窗口的后200步真实值和预测值已保存到 three_windows_prediction.csv')

# 说明：
# PatchTST等Transformer模型采用注意力机制，预测未来序列时，模型并不是简单地“延续”输入序列的最后一个点，而是综合整个输入窗口的信息（如周期、趋势、变化模式），对未来序列进行整体预测。
# 因此，预测序列的第一个点不一定与输入序列的最后一个点直接相连，而是模型根据历史窗口整体特征推断出的未来走势。这也是注意力机制的优势：可以捕捉长距离依赖和复杂时序关系。

