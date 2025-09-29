#!/bin/bash
# Custom Dataset PatchTST Training Script
# Usage: bash custom_dataset.sh

# Create log directories
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# ===========================================
# Dataset Configuration - 请根据您的数据集修改这些参数
# ===========================================
seq_len=96                           # 输入序列长度
model_name=PatchTST                  # 模型名称

# 数据路径配置
root_path_name=./dataset/            # 数据集根目录
data_path_name=your_data.csv         # 您的数据文件名
model_id_name=CustomData             # 模型标识名称
data_name=custom                     # 数据集类型名称（用于data_factory.py中的映射）

# 模型超参数配置
enc_in=7                            # 输入特征数量（请根据您的数据列数修改）
d_model=16                          # 模型维度
d_ff=128                            # 前馈网络维度
e_layers=3                          # 编码器层数
n_heads=4                           # 注意力头数量
dropout=0.3                         # Dropout率
fc_dropout=0.3                      # 全连接层Dropout率
head_dropout=0                      # 头部Dropout率

# PatchTST特定参数
patch_len=16                        # 补丁长度
stride=8                            # 步长

# 训练参数
train_epochs=100                    # 训练轮数
batch_size=128                      # 批次大小
learning_rate=0.0001               # 学习率
random_seed=2021                    # 随机种子

# ===========================================
# 预测长度循环 - 可根据需要修改预测窗口
# ===========================================
for pred_len in 24 48 96 192
do
    echo "Training PatchTST for prediction length: $pred_len"
    
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in $enc_in \
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_model $d_model \
      --d_ff $d_ff \
      --dropout $dropout \
      --fc_dropout $fc_dropout \
      --head_dropout $head_dropout \
      --patch_len $patch_len \
      --stride $stride \
      --des 'Exp' \
      --train_epochs $train_epochs \
      --itr 1 \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
    
    echo "Completed training for prediction length: $pred_len"
    echo "Log saved to: logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log"
    echo "----------------------------------------"
done

echo "All experiments completed!"