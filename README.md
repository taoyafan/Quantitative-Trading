# 使用强化学习做量化交易

## 环境

python 3.6

tushare

tensorflow



## 文件说明



+ 获取数据.ipynb：获取股票历史数据，使用 tushare pro 的接口，使用需要更换 token
+ 模型训练与测试.ipynb：早期程序，包括整个环境的搭建、强化学习的架构，但是很多细节没有完善，无法工作，已经停止更新，详细程序参考 .py 文件 



+ data_set.py：数据集文件，包括数据的读取，添加，存储，和获取 batch

+ env.py：强化学习的环境文件，包括 Observation 的定义、Action 的定义，以及模拟历史交易，返回 Observation 和 Reward

+ model.py：整个模型的搭建。

  

## 模型说明

目前采用双向 LSTM 来预测未来走势，学习从 Observation 到 State的映射，policy function 和 Q function 均采用全连接来实现。
特征目前使用的只有收盘价，历史数据长度为60，每组数据规则化为每个时刻价格相对于改组起始数据的涨幅*100