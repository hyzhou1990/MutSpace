#!/bin/bash

export MSA_data_folder='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/MSA'
export MSA_list='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/mappings/rsv_mapping.csv'
export MSA_weights_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/weights'
export VAE_checkpoint_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results/VAE_parameters'
export model_name_suffix='RSV_F_model_ddp'
export model_parameters_location='/home/gpu7/Fat-48T/Work/MutSpace2/EVE/EVE/default_model_params.json'
export training_logs_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/logs/'
export protein_index=0

# 检测可用的GPU数量
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ $num_gpus -eq 0 ]; then
    # 如果nvidia-smi命令失败，使用PyTorch检测GPU
    num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
fi
echo "检测到 $num_gpus 个GPU"

# 设置性能相关的环境变量
export CUDA_LAUNCH_BLOCKING=0  # 提高CUDA启动性能
export NCCL_DEBUG=INFO  # 打印NCCL调试信息
export NCCL_SOCKET_IFNAME=^lo  # 避免使用环回接口
export NCCL_P2P_DISABLE=0  # 启用P2P通信

# 设置CPU线程数以避免过度线程竞争
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# 创建必要的目录
mkdir -p ${VAE_checkpoint_location}
mkdir -p ${training_logs_location}
mkdir -p ${MSA_weights_location}

echo "使用优化的DDP训练脚本，在所有可用GPU上进行训练..."

# 添加EVE目录到Python路径
export PYTHONPATH=$PYTHONPATH:/home/gpu7/Fat-48T/Work/MutSpace2/EVE

cd /home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE

python ddp_train_VAE.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index ${protein_index} \
    --MSA_weights_location ${MSA_weights_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --training_logs_location ${training_logs_location} \
    --seed 42 