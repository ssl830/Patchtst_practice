#!/bin/bash
# filepath: angle_dataset.sh

# 创建日志目录
mkdir -p ./logs/LongForecasting

# 配置参数
seq_len=200
model_name="PatchTST"
root_path_name="../../dataset/"
data_path_name="data.csv"
model_id_name="AngleData"
data_name="angle"
target="z_angle_1"

# 模型参数
enc_in=3
d_model=128
d_ff=256
e_layers=3
n_heads=4
dropout=0.1
fc_dropout=0.1
head_dropout=0.0

# PatchTST参数
patch_len=16
stride=8

# 训练参数
train_epochs=100
patience=20
batch_size=32
learning_rate=0.0001
random_seed=2021

# 预测长度
pred_lengths=(200)

echo "Starting PatchTST training..."

for pred_len in "${pred_lengths[@]}"
do
    model_id="${model_id_name}_${seq_len}_${pred_len}"
    log_file="logs/LongForecasting/${model_name}_${model_id}.log"

    python ../../run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in $enc_in \
      --dec_in $enc_in \
      --c_out $enc_in \
      --e_layers $e_layers \
      --d_layers 1 \
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
      --patience $patience \
      --itr 1 \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --target $target | tee $log_file

    if [ $? -eq 0 ]; then
        echo "✅ Training completed: $log_file"
    else
        echo "❌ Training failed: $log_file"
    fi
done