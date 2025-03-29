# utils.py

import torch

def mutual_information_loss(z):
    """
    计算互信息损失：示例实现，具体计算方法需要根据论文公式进行调整
    z: 节点的隐变量，形状为 [num_nodes, latent_dim]
    """
    # 示例：可以计算 z 的协方差矩阵，并构造损失
    cov = torch.mm(z.t(), z) / (z.size(0) - 1)
    # 让协方差矩阵越接近单位矩阵越好
    identity = torch.eye(cov.size(0), device=cov.device)
    mi_loss = torch.norm(cov - identity)
    return mi_loss
