import pandas as pd

def generate_graph_from_csv(input_csv, node_csv, edge_csv):
    """
    将原始流量 CSV 转成 node_features.csv 和 edge_list.csv
    每行流量视为一个节点，若两条流量的源IP相同，则加一条边。
    """
    # 1. 读取原始 CSV
    df = pd.read_csv(input_csv)
    
    # 清理列名中的空格
    df.columns = df.columns.str.strip()
    
    # 打印列名以确认
    print("CSV 文件的列名:", df.columns)
    
    # 检查必要的列是否存在
    required_columns = ['Flow Duration', 'Destination Port', 'Label']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"缺少必要的列: {col}")
    
    # 2. 生成 node_features.csv
    df_node = pd.DataFrame()
    df_node['node_id'] = df.index  # 行号做 node_id
    df_node['Src_IP'] = df['Flow Duration']  # 源 IP
    df_node['Dst_IP'] = df['Destination Port']  # 目标 IP
    df_node['Label'] = df['Label']  # 标签
    
    # 保存到 node_csv
    df_node.to_csv(node_csv, index=False)
    print(f"已生成节点文件: {node_csv}")
    
    # 3. 生成 edge_list.csv
    #   - 规则：如果两条流(行)的 Source IP 相同，就加一条无向边
    src_ip_map = {}
    for idx, row in df_node.iterrows():
        src_ip = row['Src_IP']
        node_id = row['node_id']
        if src_ip not in src_ip_map:
            src_ip_map[src_ip] = []
        src_ip_map[src_ip].append(node_id)
    
    # 根据映射生成边
    edge_list = []
    for ip, nodes in src_ip_map.items():
        n = len(nodes)
        if n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    edge_list.append((nodes[i], nodes[j]))
    
    # 转成 DataFrame
    df_edge = pd.DataFrame(edge_list, columns=['source_node_id', 'target_node_id'])
    
    # 保存到 edge_csv
    df_edge.to_csv(edge_csv, index=False)
    print(f"已生成边文件: {edge_csv}")


if __name__ == "__main__":
    # 你要处理的原始 CSV 文件名
    input_csv = r"D:\ContestProject\Model\DataBase\MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv"
    
    # 输出的节点文件和边文件
    node_csv = "node_features.csv"
    edge_csv = "edge_list.csv"
    
    # 调用函数
    generate_graph_from_csv(input_csv, node_csv, edge_csv)
    
    print("转换完成！")