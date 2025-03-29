# data_loader.py

import os
import torch
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np

def load_cicids_data(data_path):
    """
    假设数据为 CSV 格式，需要根据实际情况进行调整。
    这里给出一个简单示例：从 CSV 中加载节点特征及边信息。
    """
    # 示例：读取节点特征文件和边列表文件
    node_features = pd.read_csv(os.path.join(data_path, 'node_features.csv'))
    edge_index = pd.read_csv(os.path.join(data_path, 'edge_list.csv'))
    
    # 将数据转换为 tensor
    x = torch.tensor(node_features.values, dtype=torch.float)
    edge_index = torch.tensor(edge_index.values.T, dtype=torch.long)  # PyG要求形状为 [2, num_edges]
    
    # 此处假设每个节点的标签（例如威胁分数，后续可用于监督训练）也在数据中
    if 'label' in node_features.columns:
        y = torch.tensor(node_features['label'].values, dtype=torch.long)
    else:
        y = None

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def get_data_loaders(data_path, batch_size):
    data = load_cicids_data(data_path)
    # 对于图数据，可以直接用 DataLoader 进行批量处理（如果图较大，也可以考虑分割）
    dataset = [data]  # 如果数据集中只有一张图，则包装成列表
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

if __name__ == '__main__':
    # 测试数据加载
    from config import DATA_PATH, BATCH_SIZE
    loader = get_data_loaders(DATA_PATH, BATCH_SIZE)
    for batch in loader:
        print(batch)
