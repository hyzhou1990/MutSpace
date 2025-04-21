#!/bin/bash

# Set log directory
LOG_DIR="/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/sequential_logs"
mkdir -p $LOG_DIR

# Record start time
echo "Starting complete analysis workflow: $(date)" > $LOG_DIR/full_process.log

# Step 1: Train VAE model
echo "Starting Step 1: Training VAE model - $(date)" | tee -a $LOG_DIR/full_process.log
cd /home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE
chmod +x run_EVE_ddp_step1_train_VAE.sh
./run_EVE_ddp_step1_train_VAE.sh > $LOG_DIR/step1.log 2>&1
echo "Completed Step 1: Training VAE model - $(date)" | tee -a $LOG_DIR/full_process.log

# Step 2: Calculate evolutionary indices
echo "Starting Step 2: Calculating evolutionary indices - $(date)" | tee -a $LOG_DIR/full_process.log
cd /home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE
chmod +x run_EVE_ddp_step2_compute_evol_indices.sh
./run_EVE_ddp_step2_compute_evol_indices.sh > $LOG_DIR/step2.log 2>&1
echo "Completed Step 2: Calculating evolutionary indices - $(date)" | tee -a $LOG_DIR/full_process.log

# Step 3: Calculate EVE scores
echo "Starting Step 3: Calculating EVE scores - $(date)" | tee -a $LOG_DIR/full_process.log
cd /home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE
chmod +x run_EVE_step3_compute_scores.sh
./run_EVE_step3_compute_scores.sh > $LOG_DIR/step3.log 2>&1
echo "Completed Step 3: Calculating EVE scores - $(date)" | tee -a $LOG_DIR/full_process.log

# Record completion time
echo "Complete workflow finished: $(date)" | tee -a $LOG_DIR/full_process.log 