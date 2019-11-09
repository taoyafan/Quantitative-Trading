import pandas as pd
import numpy as np


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
            self._history_data, self._hps.days) for x in rand_idx])
        
        next_obs = np.vstack([self.obs_buffer[x + 1].values(
            self._history_data, self._hps.days) for x in rand_idx])
        
        rewards = [self.reward_buffer[x] for x in rand_idx]
        
        actions = [self.action_buffer[x] for x in rand_idx]
        
        return obs, next_obs, rewards, actions
    
    def get_price_batch(self, nums):
        assert 1 < self._length < self._history_data.shape[0]-49, 'Length of data is {} which is not enough. \
        Data need at least {}'.format(self._length, 2)
        
        rand_idx = np.random.randint(0, self._length - 1, nums)
        close = np.array([self._history_data['close'].iloc[self.obs_buffer[x].index] for x in rand_idx])
# Debug
#         print('close:\n', close)
        obs = np.vstack([self.obs_buffer[x].values(
            self._history_data, self._hps.encode_step) for x in rand_idx])
        price_next_day = np.array([self._history_data['close'].iloc[self.obs_buffer[x].index-1]
                                   for x in rand_idx])
        temp = price_next_day - close
        up_down_prob = np.zeros([nums, 2])
        up_down_prob[np.where(temp > 0), 0] = 1
        up_down_prob[np.where(temp <= 0), 1] = 1
        return obs, up_down_prob
    
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
