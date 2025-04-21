#!/bin/bash

cd /home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE

export MSA_data_folder='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/MSA'
export MSA_list='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/mappings/rsv_mapping.csv'
export MSA_weights_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/weights'
export VAE_checkpoint_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results/VAE_parameters'
export model_name_suffix='RSV_F_model_ddp_final'
export model_parameters_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/EVE/default_model_params.json'
export training_logs_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/logs/'
export protein_index=0

export computation_mode="all_singles"
export all_singles_mutations_folder='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/mutations'
export output_evol_indices_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results/evol_indices'
export num_samples_compute_evol_indices=20000
# 增加batch_size以充分利用V100的大显存
export batch_size=4096

# 检测可用的GPU数量
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $num_gpus 个GPU"

# 如果有多个GPU，则使用DDP
if [ $num_gpus -gt 1 ]; then
    echo "将使用 $num_gpus 个GPU进行分布式计算"
    echo "启动分布式计算进化指数..."
    python ddp_compute_evol_indices.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index ${protein_index} \
    --MSA_weights_location ${MSA_weights_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --computation_mode ${computation_mode} \
    --all_singles_mutations_folder ${all_singles_mutations_folder} \
    --output_evol_indices_location ${output_evol_indices_location} \
    --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
    --batch_size ${batch_size} \
    --world_size ${num_gpus}
else
    echo "只有1个GPU可用，将使用单GPU模式计算"
    # 这里可以添加单GPU模式的代码
fi 