## 测试数据下载地址
https://www.unb.ca/cic/datasets/ids-2017.html
*进入后在页面最下面点击下载-CSVs-选择MachineLearningCSV.zip下载*

## 关于程序部分

*程序中涉及到的文件路径需要根据自己的电脑实际路径进行替换*

**gengrate_graph.py**用于处理原始数据并生成节点图和边图的程序,运行该文件后会生成相应的`node_features.csv`和`edge_list.csv`
*边文件生成部分可能占用内存过大需分多部分进行*

**data_loader.py**用于读取generate_graph.py生成的节点特征文件和边列表文件,对数据进行加载和预处理

**config.py**中定义了一些相关配置和参数

**main.py**中进行解析命令行参数并启动训练

**model.py**中定义了有关模型的函数

**train.py**中是对一些模型的初始化和计算程序

**utils.py**用于计算互信息损失
*具体计算方法还需要根据实际情况进行调整*

*后面的model.py和main.py等文件我还没有测试过，可能会出现报错需要进行调试*

