if (-not (Test-Path "./logs")) { New-Item -ItemType Directory -Path "./logs" }
if (-not (Test-Path "./logs/LongForecasting")) { New-Item -ItemType Directory -Path "./logs/LongForecasting" }

# Configuration
$seq_len = 200
$model_name = "PatchTST"
$root_path_name = "../../dataset/"
$data_path_name = "data.csv"
$model_id_name = "AngleData"
$data_name = "angle"
$target = "z_angle_1"  # 主要的预测目标列

# Model parameters (optimized)
$enc_in = 3              # 输入特征数量
$d_model = 128           # 模型维度
$d_ff = 256             # 前馈网络中的维度，在三个encoder层的前馈神经网络中使用
$e_layers = 3            # encoders层数
$n_heads = 4             # 头数
$dropout = 0.1     # Dropout率
$fc_dropout = 0.1         # 全连接层Dropout率
$head_dropout = 0.0        # 头部Dropout率

# PatchTST parameters (optimized)
$patch_len = 16
$stride = 8

# Training parameters (optimized)
$train_epochs = 100
$patience = 20
$batch_size = 32
$learning_rate = 0.00001
$random_seed = 2021

$pred_lengths = @(200)

Write-Host "Starting PatchTST training..." -ForegroundColor Green

foreach ($pred_len in $pred_lengths) {
    $model_id = "$model_id_name" + "_" + "$seq_len" + "_" + "$pred_len"
    $log_file = "logs/LongForecasting/$model_name" + "_" + "$model_id.log"

    python -u ..\..\run_longExp.py `
      --random_seed $random_seed `
      --is_training 1 `
      --root_path $root_path_name `
      --data_path $data_path_name `
      --model_id $model_id `
      --model $model_name `
      --data $data_name `
      --features M `
      --seq_len $seq_len `
      --label_len 48 `
      --pred_len $pred_len `
      --enc_in $enc_in `
      --dec_in $enc_in `
      --c_out $enc_in `
      --e_layers $e_layers `
      --d_layers 1 `
      --n_heads $n_heads `
      --d_model $d_model `
      --d_ff $d_ff `
      --dropout $dropout `
      --fc_dropout $fc_dropout `
      --head_dropout $head_dropout `
      --patch_len $patch_len `
      --stride $stride `
      --des 'Exp' `
      --train_epochs $train_epochs `
      --patience $patience `
      --itr 1 `
      --batch_size $batch_size `
      --learning_rate $learning_rate `
      --target $target | Tee-Object -FilePath $log_file
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Training completed: $log_file" -ForegroundColor Green
    } else {
        Write-Host "❌ Training failed: $log_file" -ForegroundColor Red
    }
}