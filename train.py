# train.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from config import *
from data_loader import get_data_loaders
from models import GCNModel, GATModel, DGC_VAE
from utils import mutual_information_loss  # 你可以在 utils.py 中实现计算互信息损失

def train():
    loader = get_data_loaders(DATA_PATH, BATCH_SIZE)
    device = torch.device(DEVICE)

    # 假设节点特征维度可以从数据中获得，这里以16为例
    in_channels = 16
    num_classes = 2  # 根据任务设置

    # 初始化三个模型
    model_gcn = GCNModel(in_channels, GCN_HIDDEN_DIM, num_classes).to(device)
    model_gat = GATModel(in_channels, GAT_HIDDEN_DIM, num_classes, heads=4).to(device)
    model_vae = DGC_VAE(in_channels, GCN_HIDDEN_DIM, LATENT_DIM).to(device)

    # 优化器（你可以将三个模型参数合并训练，或者分别训练）
    optimizer = torch.optim.Adam(list(model_gcn.parameters()) + 
                                 list(model_gat.parameters()) +
                                 list(model_vae.parameters()), lr=LEARNING_RATE)

    model_gcn.train()
    model_gat.train()
    model_vae.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 前向传播：分别计算各模型输出
            out_gcn = model_gcn(batch.x, batch.edge_index)
            out_gat = model_gat(batch.x, batch.edge_index)
            recon, mu, logvar = model_vae(batch.x, batch.edge_index)
            
            # 计算重构损失（例如二值交叉熵）
            recon_loss = F.binary_cross_entropy(recon, torch.matmul(batch.x, batch.x.t()))
            # KL散度
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            # 计算互信息损失（你需要根据论文公式实现具体细节）
            mi_loss = mutual_information_loss(mu)  # 这是一个示例函数
            
            # 安全评分S(t) 可由 f、g 组合而成，这里可以简单用两者输出求平均作为示例
            security_score = (out_gcn.mean() + out_gat.mean()) / 2.0
            
            # 总损失：按照论文公式
            loss = recon_loss + kl_loss + LAMBDA_MI * mi_loss + LAMBDA_S * security_score
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss:.4f}")

if __name__ == '__main__':
    train()
