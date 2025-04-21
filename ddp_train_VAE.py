import os, sys
import argparse
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from torch.cuda.amp import autocast, GradScaler

sys.path.append('/home/gpu7/Fat-48T/Work/MutSpace2/EVE')
from EVE import VAE_model
from utils import data_utils

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 设置CUDA优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 初始化进程组，使用NCCL后端
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def train(rank, world_size, args):
    """在指定的GPU上训练模型"""
    # 初始化分布式环境
    setup(rank, world_size)
    
    # 设置当前设备
    torch.cuda.set_device(rank)
    
    # 设置线程数，避免CPU过载
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    
    # 记录开始时间
    start_time = time.time()
    
    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    
    if rank == 0:
        print(f"Protein name: {protein_name}")
        print(f"MSA file: {msa_location}")
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
        print(f"Theta MSA re-weighting: {theta}")

    # 加载数据
    data = data_utils.MSA_processing(
            MSA_location=msa_location,
            theta=theta,
            use_weights=True,
            weights_location=args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    )

    model_name = protein_name + "_" + args.model_name_suffix
    
    if rank == 0:
        print(f"Model name: {model_name}")

    model_params = json.load(open(args.model_parameters_location))

    # 修改训练参数，优化多GPU效率
    if world_size > 1:
        # 增大batch_size
        original_batch_size = model_params["training_parameters"]['batch_size']
        model_params["training_parameters"]['batch_size'] = min(original_batch_size * world_size, 1024)
        
        # 如果显存足够，可以增加隐藏层维度
        if torch.cuda.get_device_properties(rank).total_memory > 30 * (1024**3):  # 显存超过30GB
            # 正确使用hidden_layers_sizes参数
            if 'hidden_layers_sizes' in model_params["encoder_parameters"]:
                # 增加隐藏层大小，但最大不超过2048
                encoder_hidden_sizes = model_params["encoder_parameters"]['hidden_layers_sizes']
                decoder_hidden_sizes = model_params["decoder_parameters"]['hidden_layers_sizes']
                
                # 增大第一层和第二层的大小
                if len(encoder_hidden_sizes) >= 2:
                    encoder_hidden_sizes[0] = min(encoder_hidden_sizes[0] * 2, 2048)
                    encoder_hidden_sizes[1] = min(encoder_hidden_sizes[1] * 2, 2048)
                    
                    # 同步修改解码器对应的层
                    if len(decoder_hidden_sizes) >= 2:
                        decoder_hidden_sizes[-1] = encoder_hidden_sizes[0]
                        decoder_hidden_sizes[-2] = encoder_hidden_sizes[1]
        
        if rank == 0:
            print(f"增大batch_size至: {model_params['training_parameters']['batch_size']}")
            print(f"隐藏层维度: {model_params['encoder_parameters']['hidden_layers_sizes']}")

    # 创建模型实例
    model = VAE_model.VAE_model(
                    model_name=model_name,
                    data=data,
                    encoder_parameters=model_params["encoder_parameters"],
                    decoder_parameters=model_params["decoder_parameters"],
                    random_seed=args.seed
    )
    
    # 将模型移到当前GPU
    model = model.to(rank)
    
    # 使用DDP包装模型
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if rank == 0:
        print(f"使用DistributedDataParallel在{world_size}个GPU上进行训练")
        print(f"模型参数总量: {sum(p.numel() for p in ddp_model.parameters())}")

    model_params["training_parameters"]['training_logs_location'] = args.training_logs_location
    model_params["training_parameters"]['model_checkpoint_location'] = args.VAE_checkpoint_location

    if rank == 0:
        print(f"Starting to train model: {model_name}")
    
    # 使用混合精度训练
    scaler = GradScaler()
    
    # 初始化Adam优化器
    optimizer = torch.optim.Adam(ddp_model.parameters(), 
                                lr=model_params["training_parameters"]['learning_rate'], 
                                weight_decay=model_params["training_parameters"]['l2_regularization'])
    
    # 可以添加学习率调度器以改善收敛
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True if rank == 0 else False
    )
    
    # 将训练步骤分散到各个GPU
    total_steps = model_params["training_parameters"]['num_training_steps']
    steps_per_gpu = total_steps // world_size
    model_params["training_parameters"]['num_training_steps'] = steps_per_gpu
    
    if rank == 0:
        print(f"每个GPU训练步数: {steps_per_gpu}")
        print(f"使用混合精度训练 (Automatic Mixed Precision)")
    
    # 创建目录
    os.makedirs(model_params["training_parameters"]['model_checkpoint_location'], exist_ok=True)
    os.makedirs(model_params["training_parameters"]['training_logs_location'], exist_ok=True)
    
    # 使用自定义训练循环替代原始的train_model方法，以支持混合精度训练
    # 检查train_model方法是否接受外部optimizer
    if hasattr(model, 'train_model_with_optimizer'):
        # 正常训练，DDP会自动同步梯度，使用自定义的训练方法
        model.train_model_with_optimizer(
            data=data, 
            training_parameters=model_params["training_parameters"],
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler
        )
    else:
        # 标准训练方法
        model.train_model(data=data, training_parameters=model_params["training_parameters"])

    # 只保存主进程的模型
    if rank == 0:
        print(f"Saving model: {model_name}")
        model.save(model_checkpoint=model_params["training_parameters"]['model_checkpoint_location']+os.sep+model_name+"_final", 
                encoder_parameters=model_params["encoder_parameters"], 
                decoder_parameters=model_params["decoder_parameters"], 
                training_parameters=model_params["training_parameters"]
        )
        
        # 计算总训练时间
        end_time = time.time()
        training_time = end_time - start_time
        print(f"总训练时间: {training_time/60:.2f} 分钟")
        print(f"每步平均时间: {training_time/total_steps:.4f} 秒")
    
    # 清理分布式环境
    cleanup()

def patch_vae_model():
    """
    给VAE_model类打补丁，添加支持混合精度训练的方法
    注意：这只会修改内存中的类，不会修改原始文件
    """
    # 检查是否已有该方法
    if hasattr(VAE_model.VAE_model, 'train_model_with_optimizer'):
        return
    
    # 添加新方法支持外部优化器和混合精度
    def train_model_with_optimizer(self, data, training_parameters, optimizer, scheduler=None, scaler=None):
        """支持外部优化器和混合精度训练的训练方法"""
        self.train()
        batch_size = training_parameters['batch_size']
        
        # 训练循环
        for training_step in range(training_parameters['num_training_steps']):
            # 获取一个随机批次
            batch, batch_seq_lens = data.get_batch(batch_size)
            batch = batch.to(self.device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 使用混合精度训练
            if scaler is not None:
                with autocast():
                    # 前向传播
                    prediction_logits, KLD = self.forward(x=batch, lengths=batch_seq_lens)
                    # 计算损失
                    reconstruction_loss = self._compute_reconstruction_loss(prediction_logits, batch)
                    loss = reconstruction_loss + KLD
                
                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 前向传播
                prediction_logits, KLD = self.forward(x=batch, lengths=batch_seq_lens)
                # 计算损失
                reconstruction_loss = self._compute_reconstruction_loss(prediction_logits, batch)
                loss = reconstruction_loss + KLD
                
                # 反向传播
                loss.backward()
                optimizer.step()
            
            # 学习率调度
            if scheduler is not None and training_step % 10 == 0:
                scheduler.step(loss)
            
            # 日志记录（只由主进程处理）
            if self.device == 0 or self.device == 'cuda:0' or self.device == torch.device('cuda:0'):
                if training_step % training_parameters['logging_frequency'] == 0:
                    self._write_training_logs(
                        training_step=training_step, 
                        loss=loss.item(), 
                        reconstruction_loss=reconstruction_loss.item(), 
                        KLD=KLD.item(), 
                        logs_location=training_parameters['training_logs_location']
                    )
                
                # 保存检查点
                if training_parameters.get('checkpoint_frequency') and training_step > 0 and \
                   training_step % training_parameters['checkpoint_frequency'] == 0:
                    self.save(
                        model_checkpoint=training_parameters['model_checkpoint_location'] + os.sep + self.model_name, 
                        encoder_parameters=self.encoder_parameters,
                        decoder_parameters=self.decoder_parameters,
                        training_parameters=training_parameters
                    )
    
    # 将方法添加到类
    VAE_model.VAE_model.train_model_with_optimizer = train_model_with_optimizer

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='VAE with DDP')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name will be the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--training_logs_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # 打补丁给VAE_model类添加新的训练方法
    patch_vae_model()
    
    # 获取可用GPU数量
    world_size = torch.cuda.device_count()
    print(f"发现 {world_size} 个GPU")
    
    if world_size > 1:
        # 创建必要的目录
        os.makedirs(args.VAE_checkpoint_location, exist_ok=True)
        os.makedirs(args.training_logs_location, exist_ok=True)
        
        # 优先使用全部4个V100
        world_size = min(world_size, 4)
        print(f"将使用 {world_size} 个GPU进行分布式训练")
        
        # 使用spawn方法启动多进程
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("只有一个GPU可用，无法使用DDP。请使用单GPU训练。")
        sys.exit(1) 