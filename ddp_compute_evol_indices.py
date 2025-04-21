import os,sys

# 添加EVE模块路径
EVE_PATH = '/home/gpu7/Fat-48T/Work/MutSpace2/EVE'
sys.path.append(EVE_PATH)

import json
import argparse
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
from tqdm import tqdm

from EVE import VAE_model
from utils import data_utils

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 使用NCCL后端，针对NVIDIA GPU优化
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 为提高性能设置CUDA优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def cleanup():
    dist.destroy_process_group()

def compute_indices_ddp(rank, world_size, args):
    # 初始化进程组
    setup(rank, world_size)
    
    # 确保所有进程使用相同的设备
    torch.cuda.set_device(rank)
    
    # 记录开始时间
    start_time = time.time()
    
    # 设置Python多线程数量，避免CPU过载
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    
    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    if rank == 0:
        print("Protein name: "+str(protein_name))
        print("MSA file: "+str(msa_location))
        print(f"CUDA设备: {torch.cuda.get_device_name(rank)}")
        print(f"可用显存: {torch.cuda.get_device_properties(rank).total_memory / (1024**3):.2f} GB")

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
    else:
        try:
            theta = float(mapping_file['theta'][args.protein_index])
        except:
            theta = 0.2
    if rank == 0:
        print("Theta MSA re-weighting: "+str(theta))

    data = data_utils.MSA_processing(
            MSA_location=msa_location,
            theta=theta,
            use_weights=True,
            weights_location=args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    )
    
    if args.computation_mode=="all_singles":
        if rank == 0:
            data.save_all_singles(output_filename=args.all_singles_mutations_folder + os.sep + protein_name + "_all_singles.csv")
        # 确保所有进程同步
        dist.barrier()
        args.mutations_location = args.all_singles_mutations_folder + os.sep + protein_name + "_all_singles.csv"
    else:
        args.mutations_location = args.mutations_location + os.sep + protein_name + ".csv"
        
    model_name = protein_name + "_" + args.model_name_suffix
    if rank == 0:
        print("Model name: "+str(model_name))

    model_params = json.load(open(args.model_parameters_location))

    model = VAE_model.VAE_model(
                    model_name=model_name,
                    data=data,
                    encoder_parameters=model_params["encoder_parameters"],
                    decoder_parameters=model_params["decoder_parameters"],
                    random_seed=42
    )
    model = model.to(rank)  # 移动模型到当前设备
    
    # 为减少内存使用，将模型设置为评估模式
    model.eval()

    try:
        checkpoint_name = str(args.VAE_checkpoint_location) + os.sep + model_name
        # 使用map_location将模型加载到正确的设备
        checkpoint = torch.load(checkpoint_name, map_location=f'cuda:{rank}')
        model.load_state_dict(checkpoint['model_state_dict'])
        if rank == 0:
            print(f"Initialized VAE with checkpoint '{checkpoint_name}'")
    except Exception as e:
        if rank == 0:
            print(f"Unable to locate VAE model checkpoint: {e}")
        cleanup()
        sys.exit(0)
    
    # 读取突变列表
    mutations_df = pd.read_csv(args.mutations_location)
    mutations_list = mutations_df['mutations'].tolist()
    
    # 根据进程数分配突变列表
    n_mutations = len(mutations_list)
    mutations_per_gpu = n_mutations // world_size
    start_idx = rank * mutations_per_gpu
    end_idx = start_idx + mutations_per_gpu if rank < world_size - 1 else n_mutations
    
    local_mutations = mutations_list[start_idx:end_idx]
    
    if rank == 0:
        print(f"Total mutations: {n_mutations}, distributed across {world_size} GPUs")
    print(f"GPU {rank} processing {len(local_mutations)} mutations")
    
    # 将本地突变列表保存到临时文件
    local_mutations_file = f"{args.all_singles_mutations_folder}/{protein_name}_gpu{rank}_mutations.csv"
    pd.DataFrame({
        'protein_name': [protein_name] * len(local_mutations),
        'mutations': local_mutations
    }).to_csv(local_mutations_file, index=False)
    
    # 计算本地突变的进化指数
    with torch.cuda.amp.autocast(enabled=True):  # 使用混合精度计算加速
        list_valid_mutations, evol_indices, _, _ = model.compute_evol_indices(
            msa_data=data,
            list_mutations_location=local_mutations_file, 
            num_samples=args.num_samples_compute_evol_indices,
            batch_size=args.batch_size
        )

    # 创建本地结果DataFrame
    local_df = pd.DataFrame({
        'protein_name': protein_name,
        'mutations': list_valid_mutations,
        'evol_indices': evol_indices
    })
    
    # 保存本地结果到临时文件
    local_output_file = f"{args.output_evol_indices_location}/{protein_name}_gpu{rank}_results.csv"
    local_df.to_csv(local_output_file, index=False)
    
    # 同步所有进程
    dist.barrier()
    
    # 在rank=0的进程上合并所有结果
    if rank == 0:
        # 合并所有GPU的结果
        all_results = []
        for r in range(world_size):
            result_file = f"{args.output_evol_indices_location}/{protein_name}_gpu{r}_results.csv"
            if os.path.exists(result_file):
                gpu_results = pd.read_csv(result_file)
                all_results.append(gpu_results)
        
        # 合并所有结果
        final_df = pd.concat(all_results, ignore_index=True)
        
        # 保存最终结果
        evol_indices_output_filename = args.output_evol_indices_location+os.sep+protein_name+'_'+str(args.num_samples_compute_evol_indices)+'_samples'+args.output_evol_indices_filename_suffix+'.csv'
        try:
            keep_header = os.stat(evol_indices_output_filename).st_size == 0
        except:
            keep_header=True 
        final_df.to_csv(path_or_buf=evol_indices_output_filename, index=False, mode='a', header=keep_header)
        
        # 计算总用时
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Combined results from all GPUs saved to {evol_indices_output_filename}")
        print(f"总共处理了 {len(final_df)} 个突变")
        print(f"总用时: {elapsed_time:.2f} 秒，平均每个突变 {elapsed_time/len(final_df)*1000:.2f} 毫秒")
        
        # 清理临时文件
        for r in range(world_size):
            result_file = f"{args.output_evol_indices_location}/{protein_name}_gpu{r}_results.csv"
            mutation_file = f"{args.all_singles_mutations_folder}/{protein_name}_gpu{r}_mutations.csv"
            if os.path.exists(result_file):
                os.remove(result_file)
            if os.path.exists(mutation_file):
                os.remove(mutation_file)
    
    # 清理分布式环境
    cleanup()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evol indices with DDP')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name is the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--computation_mode', type=str, help='Computes evol indices for all single AA mutations or for a passed in list of mutations (singles or multiples) [all_singles,input_mutations_list]')
    parser.add_argument('--all_singles_mutations_folder', type=str, help='Location for the list of generated single AA mutations')
    parser.add_argument('--mutations_location', type=str, help='Location of all mutations to compute the evol indices for')
    parser.add_argument('--output_evol_indices_location', type=str, help='Output location of computed evol indices')
    parser.add_argument('--output_evol_indices_filename_suffix', default='', type=str, help='(Optional) Suffix to be added to output filename')
    parser.add_argument('--num_samples_compute_evol_indices', type=int, help='Num of samples to approximate delta elbo when computing evol indices')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size when computing evol indices')
    parser.add_argument('--world_size', default=2, type=int, help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs(args.output_evol_indices_location, exist_ok=True)
    
    # 启动多进程
    world_size = min(torch.cuda.device_count(), args.world_size)
    mp.spawn(compute_indices_ddp, args=(world_size, args,), nprocs=world_size, join=True) 