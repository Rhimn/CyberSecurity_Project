# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# GCN模型，用于函数 f(A, X) = softmax(GCN(A, X))
class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)  # 输出节点类别的概率分布

# GAT模型，用于函数 g(A, X) = softmax(GAT(A, X))
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)

# DGC-VAE模型：包含编码器和解码器，用于重构输入及学习异常特征
class DGC_VAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(DGC_VAE, self).__init__()
        # 编码器
        self.gc1 = GCNConv(in_channels, hidden_channels)
        self.gc_mu = GCNConv(hidden_channels, latent_dim)
        self.gc_logvar = GCNConv(hidden_channels, latent_dim)
        
        # 解码器：这里简单使用内积重构
        # 也可以设计一个更复杂的解码器网络
       
    def encode(self, x, edge_index):
        h = F.relu(self.gc1(x, edge_index))
        mu = self.gc_mu(h, edge_index)
        logvar = self.gc_logvar(h, edge_index)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        # 简单使用内积解码
        adj_recon = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_recon
    
    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

if __name__ == '__main__':
    # 测试模型前向传播
    import torch_geometric
    from torch_geometric.data import Data
    # 随机生成简单图
    x = torch.randn((100, 16))
    edge_index = torch.randint(0, 100, (2, 200))
    data = Data(x=x, edge_index=edge_index)
    
    gcn = GCNModel(in_channels=16, hidden_channels=64, out_channels=2)
    gat = GATModel(in_channels=16, hidden_channels=64, out_channels=2, heads=4)
    vae = DGC_VAE(in_channels=16, hidden_channels=64, latent_dim=32)
    
    print("GCN output:", gcn(data.x, data.edge_index).shape)
    print("GAT output:", gat(data.x, data.edge_index).shape)
    recon, mu, logvar = vae(data.x, data.edge_index)
    print("VAE recon shape:", recon.shape)
