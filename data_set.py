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
        self._history_data = pd.read_csv(data_dir, index_col=0).iloc[-1::-1, :]
        self._history_data.reset_index(drop=True, inplace=True)
        self.data_nom()
        
        return
    
    def data_nom(self):
        # 计算均线
        self._history_data['5'] = self._history_data.close.rolling(5).mean()
        self._history_data['10'] = self._history_data.close.rolling(20).mean()
        self._history_data['30'] = self._history_data.close.rolling(30).mean()
        self._history_data['60'] = self._history_data.close.rolling(60).mean()
        self._history_data['vol60'] = self._history_data.vol.rolling(60).mean()
        self._history_data.drop(range(0, 59), inplace=True)
        self._history_data.reset_index(drop=True, inplace=True)
        
        # open 价格相对 60 均线幅度
        self._history_data['nom_open'] = (self._history_data['open'] - self._history_data['60']
                                          ) / self._history_data['60']

        # low, high, close 价格相对  open
        self._history_data[['nom_' + x for x in ['close', 'high', 'low']]] \
            = self._history_data[['close', 'high', 'low']].apply(lambda x: x - self._history_data['open'])

        # 5, 10, 30, 60 均线相对上一时刻
        self._history_data[['nom_' + x for x in ['5', '10', '30', '60']]] \
            = self._history_data[['5', '10', '30', '60']].apply(lambda x: x.diff() / x)

        # 交易量相对 交易量60 均线幅度
        self._history_data['nom_vol'] = (self._history_data['vol'] - self._history_data['vol60']
                                         ) / self._history_data['vol60']
        self._history_data['nom_vol60'] = self._history_data['vol60'].diff() / self._history_data['vol60']

        self._history_data.drop([0, 4948, 4949, 17640], inplace=True)
        self._history_data.reset_index(drop=True, inplace=True)
        
        self._history_data[[x for x in self._history_data.columns if x.startswith('nom')]] = \
            self._history_data[[x for x in self._history_data.columns if x.startswith('nom')]].apply(
            lambda x: (x - x[0: self._hps.train_data_num].mean()) / x[0: self._hps.train_data_num].std())
        
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
    pd.set_option('display.width', 1000)  # 设置字符显示宽度
    pd.set_option('display.max_columns', None)
    hps = {
        'encode_step': 60,  # 历史数据个数
        'train_data_num': 100000,  # 训练集个数
        }
    
    hps = namedtuple("HParams", hps.keys())(**hps)
    data_set = DataSet(hps)
    # data_size = 100
    # for i in range(data_size):
    #     data_set.add_data(Observations(i, 0, 0, 0), 0, 0)
    # print(data_set.get_price_batch(2))
    # print(data_set.get_price_test_batch(2))
    print(data_set.history_data.head(20))
    print(data_set.history_data.tail(20))
    print(data_set.history_data.nom_vol60.std())
    
    return


if __name__ == '__main__':
    np.set_printoptions(2)
    main()

