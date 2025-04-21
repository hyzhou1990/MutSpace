#!/bin/bash

export input_evol_indices_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results/evol_indices'
export input_evol_indices_filename_suffix='_20000_samples'
export protein_list='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/mappings/rsv_mapping.csv'
export output_eve_scores_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results/EVE_scores'
export output_eve_scores_filename_suffix='RSV_F_model'

# 使用已有的GMM模型参数
export GMM_parameter_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results/GMM_parameters/RSV_F_model'
export GMM_parameter_filename_suffix='RSV_F_model'
export protein_GMM_weight=0.3

# 设置直方图保存位置（脚本会自动添加plots_histograms/RSV_F_model子目录）
export plot_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results'

cd /home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE

echo "开始更新直方图..."
python train_GMM_and_compute_EVE_scores.py \
    --input_evol_indices_location ${input_evol_indices_location} \
    --input_evol_indices_filename_suffix ${input_evol_indices_filename_suffix} \
    --protein_list ${protein_list} \
    --output_eve_scores_location ${output_eve_scores_location} \
    --output_eve_scores_filename_suffix ${output_eve_scores_filename_suffix} \
    --GMM_parameter_location ${GMM_parameter_location} \
    --GMM_parameter_filename_suffix ${GMM_parameter_filename_suffix} \
    --load_GMM_models \
    --protein_GMM_weight ${protein_GMM_weight} \
    --plot_location ${plot_location} \
    --plot_histograms \
    --verbose

echo "直方图更新完成!"

# 检查新生成的直方图
if [ -f "${plot_location}/plots_histograms/${output_eve_scores_filename_suffix}/histogram_random_samples_${output_eve_scores_filename_suffix}_all.png" ]; then
    echo "已成功生成新直方图：${plot_location}/plots_histograms/${output_eve_scores_filename_suffix}/histogram_random_samples_${output_eve_scores_filename_suffix}_all.png"
else
    echo "警告：未找到新生成的直方图！"
fi 