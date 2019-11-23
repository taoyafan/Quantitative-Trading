import pandas as pd
import numpy as np
from env import Observations


class DataSet:
    def __init__(self, hps, data_dir='601318.SH_5min.csv'):
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        
        self._length = 0
        self._hps = hps
        self._history_data = pd.read_csv(data_dir, index_col=0)

        # max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        # self._history_data[['open', 'high', 'low', 'close', 'vol']] = \
        #     self._history_data[['open', 'high', 'low', 'close', 'vol']].apply(max_min_scaler)
        return
    
    def get_batch(self, nums):
        assert self._length > 1, 'Length of data is {} which is not enough. \
        Data need at least {}'.format(self._length, 2)
        
        rand_idx = np.random.randint(0, self._length - 1, nums)
        obs = np.vstack([self.obs_buffer[x].values(
            self._history_data, self._hps.encode_step) for x in rand_idx])
        
        next_obs = np.vstack([self.obs_buffer[x + 1].values(
            self._history_data, self._hps.encode_step) for x in rand_idx])
        
        rewards = [self.reward_buffer[x] for x in rand_idx]
        
        actions = [self.action_buffer[x] for x in rand_idx]
        
        return obs, next_obs, rewards, actions
    
    def _get_obs_price(self, obs_list):
        close = np.array([self._history_data['close'].iloc[x.index] for x in obs_list])
        price_next_day = np.array([self._history_data['close'].iloc[x.index-1] for x in obs_list])
        obs = np.vstack([x.values(self._history_data, self._hps.encode_step) for x in obs_list])

        # up_down_prob 第 0 位为 1 时为上涨，反之为下跌（持平）
        temp = price_next_day - close
        up_down_prob = np.zeros([len(obs_list), 2])
        up_down_prob[np.where(temp > 0), 0] = 1
        up_down_prob[np.where(temp <= 0), 1] = 1
        return obs, up_down_prob
    
    def get_price_batch(self, nums):
        assert 1 < self._length < self._history_data.shape[0]-49, 'Length of data is {} which is not enough. \
        Data need at least {}'.format(self._length, 2)
        
        rand_idx = np.random.randint(0, self._length - 1, nums)
        return self._get_obs_price([self.obs_buffer[x] for x in rand_idx])

    def get_price_test_batch(self, nums):
        end_id = self.history_data.shape[0] - self._hps.train_data_num - self._hps.encode_step
        rand_idx = np.random.randint(0, end_id, nums)
        rand_obs = [Observations(x, 0, 0, 0) for x in rand_idx]
        return self._get_obs_price(rand_obs)
    
    def add_data(self, obs, action, reward):
        # obs 为 Observation 类
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self._length += 1
        return

    @property
    def history_data(self):
        return self._history_data


def main():
    from collections import namedtuple

    hps = {'enc_hidden_dim': 100,
           'dec_hidden_dim': 100,
           'gamma': 0.99,
           'learning_rate': 0.001,
           'batch_size': 256,
           'encode_step': 60,  # 历史数据个数
           'encode_dim': 1,  # 特征个数：时间，开，收，高，低，量

           'train_data_num': 10000,  # 训练集个数
           'train_iter': 100000,  # 训练的 iterations
           'eval_interval': 20,  # 每次测试间隔的训练次数

           'exp_name': '60相对收盘价',  # 实验名称
           'model_dir': './model',  # 保存模型文件夹路径
           'is_retrain': False}  # 是否从头训练
    hps = namedtuple("HParams", hps.keys())(**hps)
    data_set = DataSet(hps)
    data_size = 100
    for i in range(data_size):
        data_set.add_data(Observations(i, 0, 0, 0), 0, 0)
    print(data_set.get_price_batch(2))
    print(data_set.get_price_test_batch(2))
    return


if __name__ == '__main__':
    np.set_printoptions(2)
    main()

