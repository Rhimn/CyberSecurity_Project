# train.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from preprocess.config import *
from preprocess.data_loader import get_data_loaders
from models import GCNModel, GATModel, DGC_VAE
from utils import mutual_information_loss  # 你可以在 utils.py 中实现计算互信息损失
from preprocess.data_loader import get_data_loaders
from preprocess.config import DATA_PATH, BATCH_SIZE

# 获取数据加载器
loader = get_data_loaders(DATA_PATH, BATCH_SIZE)
for batch in loader:
    print(batch)  # 打印整个 batch 对象，检查其结构

# 遍历数据加载器，获取数据批次
for batch in loader:
    print(f"x shape: {batch.x.shape}")  # 打印输入特征张量的形状



def train():
    input_dim = 5  # 输入特征维度
    hidden_dim = 16  # 隐藏层维度
    num_classes = 2  # 根据任务设置
    
    # 初始化模型
    model_gcn = GCNModel(input_dim, hidden_dim, num_classes)

    # 获取数据加载器
    loader = get_data_loaders(DATA_PATH, BATCH_SIZE)
    device = torch.device(DEVICE)

    # 初始化三个模型
    model_gcn = GCNModel(input_dim, GCN_HIDDEN_DIM, num_classes).to(device)
    model_gat = GATModel(input_dim, GAT_HIDDEN_DIM, num_classes, heads=4).to(device)
    model_vae = DGC_VAE(input_dim, GCN_HIDDEN_DIM, LATENT_DIM).to(device)

    # 优化器
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
            
            # 前向传播
            out_gcn = model_gcn(batch.x, batch.edge_index)
            out_gat = model_gat(batch.x, batch.edge_index)
            recon, mu, logvar = model_vae(batch.x, batch.edge_index)
            
            # 计算损失
            recon_loss = F.binary_cross_entropy(recon, torch.matmul(batch.x, batch.x.t()))
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            mi_loss = mutual_information_loss(mu)
            
            # 安全评分
            security_score = (out_gcn.mean() + out_gat.mean()) / 2.0
            
            # 总损失
            loss = recon_loss + kl_loss + LAMBDA_MI * mi_loss + LAMBDA_S * security_score
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss:.4f}")


if __name__ == '__main__':
    train()
