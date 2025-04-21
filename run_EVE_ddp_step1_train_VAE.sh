#!/bin/bash

export MSA_data_folder='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/MSA'
export MSA_list='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/mappings/rsv_mapping.csv'
export MSA_weights_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/weights'
export VAE_checkpoint_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results/VAE_parameters'
export model_name_suffix='RSV_F_model_ddp'
export model_parameters_location='/home/gpu7/Fat-48T/Work/MutSpace2/EVE/EVE/default_model_params.json'
export training_logs_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/logs/'
export protein_index=0

# Detect available GPU count
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ $num_gpus -eq 0 ]; then
    # If nvidia-smi command fails, use PyTorch to detect GPUs
    num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
fi
echo "Detected $num_gpus GPUs"

# Set performance-related environment variables
export CUDA_LAUNCH_BLOCKING=0  # Improve CUDA launch performance
export NCCL_DEBUG=INFO  # Print NCCL debug information
export NCCL_SOCKET_IFNAME=^lo  # Avoid using loopback interface
export NCCL_P2P_DISABLE=0  # Enable P2P communication

# Set CPU thread count to avoid excessive thread contention
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# Create necessary directories
mkdir -p ${VAE_checkpoint_location}
mkdir -p ${training_logs_location}
mkdir -p ${MSA_weights_location}

echo "Using optimized DDP training script for training on all available GPUs..."

# Add EVE directory to Python path
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