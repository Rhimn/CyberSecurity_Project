import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import LabelEncoder

def load_cicids_data(data_path):
    # 拼接节点和边文件的路径，建议使用正斜杠或原始字符串
    node_path = os.path.join(data_path, "D:\ContestProject\Model\\nodes.csv")
    edge_path = os.path.join(data_path, "D:\ContestProject\Model\edges.csv")
    
    # 读取 CSV 文件
    node_df = pd.read_csv(node_path)
    edge_df = pd.read_csv(edge_path)
    
    # 清理节点文件的列名：去除空格并统一转为小写
    node_df.columns = node_df.columns.str.strip().str.lower()
    edge_df.columns = edge_df.columns.str.strip().str.lower()
    
    print("清理后的节点文件列名：", node_df.columns.tolist())
    
    # 检查标签列是否存在
    if 'label' not in node_df.columns:
        raise KeyError("节点文件中未找到 'label' 列！")
    
    # 使用 LabelEncoder 将标签转换为数值（仅转换一次）
    label_encoder = LabelEncoder()
    node_df['label'] = label_encoder.fit_transform(node_df['label'].astype(str))
    
    # 指定用于训练的特征列（注意，这里列名已变为小写）
    feature_cols = ['flow_duration', 'total_fwd_packets', 'total_backward_packets', 'flow_bytes/s', 'flow_packets/s']
    # 提取特征数据，并填充缺失值
    features = node_df[feature_cols].fillna(0)
    # 将所有特征列转换为数值型（如果有非数值数据，将其转换为 NaN，再填充为0）
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 转换为 Tensor，类型为 float
    x = torch.tensor(features.values, dtype=torch.float)
    # 获取标签，转换为长整型 Tensor
    y = torch.tensor(node_df['label'].values, dtype=torch.long)
    
    # 处理边文件：假设边文件中包含 "source" 和 "target" 两列
    edge_index = torch.tensor(edge_df[['source', 'target']].values.T, dtype=torch.long)
    
    # 创建并返回 PyG Data 对象
    return Data(x=x, edge_index=edge_index, y=y)

def get_data_loaders(data_path, batch_size):
    data = load_cicids_data(data_path)
    # 如果数据集中只有一张图，将 Data 对象包装成列表
    loader = DataLoader([data], batch_size=batch_size, shuffle=True)
    return loader

if __name__ == '__main__':
    # 从配置文件中加载数据路径和批量大小
    from config import DATA_PATH, BATCH_SIZE
    loader = get_data_loaders(DATA_PATH, BATCH_SIZE)
    for batch in loader:
        print(batch)
