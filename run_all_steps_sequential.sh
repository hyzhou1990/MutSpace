#!/bin/bash

# 设置日志目录
LOG_DIR="/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/sequential_logs"
mkdir -p $LOG_DIR

# 记录开始时间
echo "开始全部分析流程: $(date)" > $LOG_DIR/full_process.log

# 第一步：训练VAE模型
echo "开始步骤1: 训练VAE模型 - $(date)" | tee -a $LOG_DIR/full_process.log
cd /home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE
chmod +x run_EVE_ddp_step1_train_VAE.sh
./run_EVE_ddp_step1_train_VAE.sh > $LOG_DIR/step1.log 2>&1
echo "完成步骤1: 训练VAE模型 - $(date)" | tee -a $LOG_DIR/full_process.log

# 第二步：计算进化指数
echo "开始步骤2: 计算进化指数 - $(date)" | tee -a $LOG_DIR/full_process.log
cd /home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE
chmod +x run_EVE_ddp_step2_compute_evol_indices.sh
./run_EVE_ddp_step2_compute_evol_indices.sh > $LOG_DIR/step2.log 2>&1
echo "完成步骤2: 计算进化指数 - $(date)" | tee -a $LOG_DIR/full_process.log

# 第三步：计算EVE得分
echo "开始步骤3: 计算EVE得分 - $(date)" | tee -a $LOG_DIR/full_process.log
cd /home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE
chmod +x run_EVE_step3_compute_scores.sh
./run_EVE_step3_compute_scores.sh > $LOG_DIR/step3.log 2>&1
echo "完成步骤3: 计算EVE得分 - $(date)" | tee -a $LOG_DIR/full_process.log

# 记录完成时间
echo "全部流程完成: $(date)" | tee -a $LOG_DIR/full_process.log 