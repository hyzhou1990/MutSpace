import os
import pandas as pd
from Bio import SeqIO

# 读取FASTA序列文件
fasta_file = '/home/gpu7/Fat-48T/Work/MutSpace2/RSV.fasta'
output_csv = '/home/gpu7/Fat-48T/Work/MutSpace2/RSV_EVE/mutations/RSV_F_mutations.csv'

# 所有标准氨基酸
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

# 读取FASTA序列
records = list(SeqIO.parse(fasta_file, "fasta"))
if len(records) > 0:
    wt_seq = str(records[0].seq)
    print(f"序列长度: {len(wt_seq)}")
else:
    print("无法读取序列文件")
    exit(1)

# 生成所有可能的单氨基酸突变
mutations = []
for i, wt_aa in enumerate(wt_seq, 1):
    for mut_aa in amino_acids:
        if wt_aa != mut_aa:  # 跳过不变的情况
            mutation = f"{wt_aa}{i}{mut_aa}"
            mutations.append({"protein_name": "RSV_F", "mutations": mutation})

# 创建DataFrame并保存为CSV
df = pd.DataFrame(mutations)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)

print(f"生成了 {len(mutations)} 个突变，已保存到 {output_csv}") 