# config.py

DATA_PATH = 'D:\\ContestProject\\Model'  # 存放预处理文件的目录
BATCH_SIZE = 32 

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
