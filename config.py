# config.py

# 数据相关配置
DATA_PATH = './data/cicids2017w/'  # 数据存放路径
BATCH_SIZE = 64
NUM_WORKERS = 4

# 模型超参数
GCN_HIDDEN_DIM = 64
GAT_HIDDEN_DIM = 64
LATENT_DIM = 32  # DGC-VAE中编码器输出的维度

# 训练参数
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3

# 超参数λ（例如用于互信息项，及安全评分项λ_s）
LAMBDA_MI = 0.1
LAMBDA_S = 0.1

# 其他设置
DEVICE = 'cuda'  # 或 'cpu'
