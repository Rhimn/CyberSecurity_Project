import torch
import pandas as pd
  
data = pd.read_csv("Monday-WorkingHours.pcap_ISCX.csv")

# 删除无效列（如空值超过90%的列）
data = data.dropna(axis=1, thresh=0.9*len(data))
  # 填充剩余空值
data = data.fillna(0)

import networkx as nx
from torch_geometric.data import Data

  # 示例：以源IP和目标IP为节点，流量为边
unique_ips = pd.unique(data[[' Source IP', ' Destination IP']].values.ravel('K'))
ip_to_idx = {ip: i for i, ip in enumerate(unique_ips)}

  # 构建边索引（edge_index）
src = data[' Source IP'].map(ip_to_idx).values
dst = data[' Destination IP'].map(ip_to_idx).values
edge_index = torch.tensor([src, dst], dtype=torch.long)

  # 节点特征（示例：取流量统计特征）
features = data[[' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets']].values
x = torch.tensor(features, dtype=torch.float)

  # 标签（假设最后一列为攻击标签）
y = torch.tensor(data[' Label'].map({'BENIGN':0, 'DoS':1}).values, dtype=torch.long)

  # 转换为PyG的Data对象
graph_data = Data(x=x, edge_index=edge_index, y=y)

from torch_geometric.data import InMemoryDataset

class MyDataset(InMemoryDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        # 这里放置上述预处理代码
        data_list = [graph_data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = MyDataset(root='./data')