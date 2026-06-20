# Evoformer Autoencoder

基于AlphaFold2 Evoformer架构的单细胞RNA测序自编码器，已适配PerturbNova框架。

## 快速开始

### 在Replogle数据集（2000 HVG）上使用

```bash
# Stage 1: 训练Evoformer Autoencoder
python -m perturbnova.cli.train --config configs/replogle/training/stage1_evoformer.toml

# Stage 2: 使用冻结的Evoformer AE训练扩散模型
python -m perturbnova.cli.train --config configs/replogle/training/stage2.toml
```

配置文件已创建在 `configs/replogle/training/stage1_evoformer.toml`。

## 概述

Evoformer Autoencoder使用蛋白质结构预测中的Evoformer架构来学习单细胞基因表达数据的表示。该模型：

- 使用BERT风格的掩码基因预测进行预训练
- 通过三角注意力和乘法机制建模基因间的相互作用
- 可作为PerturbNova中VAE的替代方案

## 架构特点

### 核心组件

1. **MSA行注意力（带配对偏置）**: 类似于AlphaFold2中的MSA行注意力，使用pair representation作为注意力偏置
2. **MSA列注意力**: 跨细胞（或基因组）的注意力机制
3. **三角乘法**: 用于更新pair representation的三角乘法（outgoing和incoming两种模式）
4. **三角注意力**: 用于更新pair representation的三角注意力
5. **外积均值**: 从MSA representation更新pair representation

### 数据表示

- **MSA Representation**: `[batch, n_gene, n_gene_feat]` - 类似于蛋白质MSA中的序列表示
- **Pair Representation**: `[batch, n_gene, n_gene, n_pair_feat]` - 类似于AlphaFold2中的残基对表示

## 使用方法

### 1. 作为独立模型训练

```bash
# 使用conda环境
conda activate my_state

# 训练Evoformer Autoencoder
python scripts/train_evoformer_ae.py \
    --data_path /path/to/data.h5ad \
    --output_dir ./outputs/evoformer_ae \
    --n_gene_total 20074 \
    --n_gene 100 \
    --n_gene_feat 32 \
    --n_pair_feat 16 \
    --n_embed 1280 \
    --num_evoformer_blocks 6 \
    --latent_dim 128 \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4
```

### 2. 集成到PerturbNova

在配置文件中启用Evoformer Autoencoder：

```toml
# configs/evoformer_ae_example.toml

[vae]
enabled = false  # 禁用标准VAE

[evoformer_ae]
enabled = true
checkpoint_path = ""  # 预训练检查点路径
latent_dim = 128
freeze = false  # 是否在扩散模型训练时冻结

# 模型架构参数
n_gene_total = 20074
n_gene = 100
n_gene_feat = 32
n_pair_feat = 16
n_embed = 1280
num_evoformer_blocks = 6

# 训练参数
reconstruction_loss_weight = 0.1
decode_predictions = true
batch_size = 512
```

### 3. 在代码中使用

```python
import torch
from perturbnova.evoformer_ae import (
    EvoformerAutoencoder,
    build_evoformer_ae_module,
    encode_with_evoformer_ae,
    decode_with_evoformer_ae,
)

# 创建模型
model = EvoformerAutoencoder(
    n_gene_total=20074,
    n_gene=100,
    n_gene_feat=32,
    n_pair_feat=16,
    n_embed=1280,
    num_evoformer_blocks=6,
    latent_dim=128,
)

# 编码
sc_data = torch.randn(4, 20074)  # [batch, n_genes]
latent = model.encode(sc_data)  # [batch, 128]

# 解码
reconstructed = model.decode(latent)  # [batch, 20074]

# 自编码器模式
output = model(sc_data, mode="autoencoder")
# output["latent"]: [batch, 128]
# output["reconstructed"]: [batch, 20074]

# 预训练模式（掩码基因预测）
output = model(sc_data, mode="pretrain")
# output["pred"]: [batch, 20074]
# output["embedding"]: [batch, 1280]
```

### 4. PerturbNova集成

```python
from perturbnova.evoformer_ae import build_evoformer_ae_module

# 配置
config = {
    "enabled": True,
    "checkpoint_path": "path/to/checkpoint.pt",
    "latent_dim": 128,
    "freeze": True,
    "n_gene_total": 20074,
    "n_gene": 100,
    "n_gene_feat": 32,
    "n_pair_feat": 16,
    "n_embed": 1280,
    "num_evoformer_blocks": 6,
}

# 构建模块
device = torch.device("cuda")
evoformer_ae = build_evoformer_ae_module(config, input_dim=20074, device=device)

# 使用
if evoformer_ae is not None:
    latent = encode_with_evoformer_ae(evoformer_ae, sc_data)
    reconstructed = decode_with_evoformer_ae(evoformer_ae, latent)
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_gene_total` | 20074 | 总基因数（输入/输出维度） |
| `n_gene` | 100 | 基因组数（MSA位置数） |
| `n_gene_feat` | 32 | 每个基因组的特征维度 |
| `n_pair_feat` | 16 | Pair representation特征维度 |
| `n_embed` | 1280 | 预测头的隐藏维度 |
| `num_evoformer_blocks` | 6 | Evoformer块数量 |
| `latent_dim` | 128 | 潜在空间维度 |

## 模型规模

使用默认参数时：
- **可训练参数**: ~1.4M
- **模型大小**: ~770MB（保存为.h5格式时）
- **GPU内存**: ~8GB（batch_size=4时）

## 与标准VAE的对比

| 特性 | 标准MLP-VAE | Evoformer Autoencoder |
|------|-------------|----------------------|
| 架构 | MLP层叠 | Evoformer块（注意力+三角操作） |
| 基因交互 | 隐式（通过全连接） | 显式（通过pair representation） |
| 参数量 | ~5M | ~1.4M |
| 训练速度 | 快 | 较慢（注意力计算） |
| 表示能力 | 局部 | 全局（基因间关系） |

## 从TensorFlow迁移权重

如果需要从原始TensorFlow版本迁移权重：

```python
import h5py
import torch

def load_tf_weights(pytorch_model, tf_checkpoint_path):
    """从TensorFlow检查点加载权重到PyTorch模型。"""
    with h5py.File(tf_checkpoint_path, 'r') as f:
        # 需要根据具体的TF权重结构进行映射
        # 这里提供一个示例框架
        for name, param in pytorch_model.named_parameters():
            # 查找对应的TF权重
            # tf_name = convert_pytorch_name_to_tf(name)
            # if tf_name in f:
            #     param.data = torch.tensor(f[tf_name][:])
            pass
    return pytorch_model
```

注意：由于TensorFlow和PyTorch的权重格式不同，需要编写自定义的权重转换脚本。

## 测试

运行单元测试：

```bash
cd /work/home/cryoem666/xyf/temp/pycharm/PerturbNova
conda activate my_state
pytest tests/test_evoformer_ae.py -v
```

## 参考文献

1. AlphaFold2: Jumper et al., "Highly accurate protein structure prediction with AlphaFold", Nature, 2021
2. 原始TensorFlow实现: `/work/home/xugang/projects/single_cell_llm/v10_mse_xfy/`

## 许可证

遵循PerturbNova项目的许可证。
