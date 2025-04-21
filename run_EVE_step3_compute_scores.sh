#!/bin/bash

export input_evol_indices_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results/evol_indices'
export input_evol_indices_filename_suffix='_20000_samples'
export protein_list='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/mappings/rsv_mapping.csv'
export output_eve_scores_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results/EVE_scores'
export output_eve_scores_filename_suffix='RSV_F_model'

# Create GMM model parameter location for RSV_F protein
mkdir -p /home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results/GMM_parameters/
export GMM_parameter_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results/GMM_parameters'
export GMM_parameter_filename_suffix='RSV_F_model'
export protein_GMM_weight=0.3
export plot_location='/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/results'

cd /home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE

echo "Starting EVE score calculation..."
python train_GMM_and_compute_EVE_scores.py \
    --input_evol_indices_location ${input_evol_indices_location} \
    --input_evol_indices_filename_suffix ${input_evol_indices_filename_suffix} \
    --protein_list ${protein_list} \
    --output_eve_scores_location ${output_eve_scores_location} \
    --output_eve_scores_filename_suffix ${output_eve_scores_filename_suffix} \
    --GMM_parameter_location ${GMM_parameter_location} \
    --GMM_parameter_filename_suffix ${GMM_parameter_filename_suffix} \
    --compute_EVE_scores \
    --protein_GMM_weight ${protein_GMM_weight} \
    --plot_location ${plot_location} \
    --verbose 

echo "EVE score calculation completed!" 