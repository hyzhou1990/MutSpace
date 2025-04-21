# RSV_EVE

呼吸道合胞病毒（RSV）的进化变异效应（EVE）评分计算工具。

## 项目概述

本项目基于EVE（Evolutionary model of Variant Effect）框架，用于分析RSV蛋白质的突变效应。通过深度学习和统计建模，计算各种氨基酸突变的进化保守性得分，评估其对蛋白质功能的潜在影响。

## 运行流程

整个分析流程分为三个主要步骤：

### 1. 训练VAE模型

```bash
chmod +x run_EVE_ddp_step1_train_VAE.sh
./run_EVE_ddp_step1_train_VAE.sh
```

该步骤使用分布式数据并行（DDP）在多个GPU上训练变分自编码器（VAE）模型，学习RSV蛋白质序列的潜在表示。

### 2. 计算进化指数

```bash
chmod +x run_EVE_ddp_step2_compute_evol_indices.sh
./run_EVE_ddp_step2_compute_evol_indices.sh
```

该步骤使用训练好的VAE模型计算每个可能突变的进化指数。

### 3. 计算EVE分数

```bash
chmod +x run_EVE_step3_compute_scores.sh
./run_EVE_step3_compute_scores.sh
```

该步骤训练高斯混合模型（GMM）并计算每个突变的EVE分数，结果保存在`results/EVE_scores/`目录中。

## 输出文件

主要输出文件位于：
- VAE模型参数：`results/VAE_parameters/`
- 进化指数：`results/evol_indices/`
- EVE分数：`results/EVE_scores/all_EVE_scores_RSV_F_model.csv`

## 依赖项

- Python 3.6+
- PyTorch 1.7+
- scikit-learn
- pandas
- numpy
- matplotlib

## 目录结构

- `MSA/`: 多序列比对文件
- `mappings/`: 蛋白质映射文件
- `results/`: 结果输出目录
- `utils/`: 工具函数
- `*.py`: 主要脚本文件
- `run_*.sh`: 运行脚本

## 注意事项

- 确保有足够的GPU内存来训练VAE模型
- 运行时间可能较长，尤其是在处理大型MSA文件时
- 所有步骤都提供详细的日志输出，保存在当前目录下 