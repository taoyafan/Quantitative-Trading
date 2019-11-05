import tensorflow as tf
from env import *
from collections import namedtuple
import os
import shutil
from data_set import DataSet
import time


class Model:
    def __init__(self, hps, obs_dim, actions_dim):
        self._hps = hps
        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.action, self.Q, self.price_predict, self.action_loss, self.Q_loss, self.price_loss, \
        self.action_train_opt, self.Q_train_opt, self.price_train_opt = self._build_graph()
        
        self._sess, self._summary_writer = self._sess_setup()
        return
    
    def policy_q_train(self, iteration, data_set):
        for i in range(iteration):
            obs, next_obs, reward = data_set.get_batch(self._hps.batch_size)
            action_loss, Q_loss = self._train_one_step(obs, next_obs, reward)
            print('action_loss: {}, Q_loss : {}'.format(action_loss, Q_loss))
        
        return
    
    def price_train(self, iteration, data_set):
        for i in range(iteration):
            obs_ph, price_ph = data_set.get_price_batch(self._hps.batch_size)
            # print('obs_ph\n', obs_ph)
            price, price_loss, _ = self._sess.run([self.price_predict, self.price_loss, self.price_train_opt],
                                                  {self._observations_ph: (obs_ph-50)/100,
                                                   self._price_next_day_ph: price_ph})
            
            print('price_predict: \n {} \n target: \n {} \n price_loss : {}'.format(price, price_ph, price_loss))
            err_rate = price * price_ph
            err_rate = err_rate <= 0
            print('Error rate : {}'.format(np.sum(err_rate) / err_rate.shape))
            
    def test(self, data):
        return
    
    def predict(self, obs):
        action_prob = self._sess.run(self.action, {self._observations_ph: obs})
        return action_prob
    
    def _train_one_step(self, obs, next_obs, reward):
        feed_dict = {self._observations_ph: obs,
                     self._next_observations_ph: next_obs,
                     self._rewards_ph: reward}
        
        action_loss, Q_loss, _, _ = self._sess.run([self.action_loss, self.Q_loss,
                                                    self.action_train_opt, self.Q_train_opt],
                                                   feed_dict)
        return action_loss, Q_loss
    
    def _sess_setup(self):
        saver = tf.train.Saver(max_to_keep=3)
        sv = tf.train.Supervisor(logdir=self._hps.train_dir,
                                 is_chief=True,
                                 saver=saver,
                                 summary_op=None,
                                 save_summaries_secs=600,  # save summaries for tensorboard every 60 secs
                                 save_model_secs=600,  # checkpoint every 600 secs
                                 global_step=self.global_step,
                                 init_feed_dict=None
                                 )
        summary_writer = sv.summary_writer
        sess = sv.prepare_or_wait_for_session()
        
        return sess, summary_writer
    
    def _create_placeholders(self):
        observations_dim = self._obs_dim
        actions_dim = self._actions_dim
        
        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observations_dim),
            name='observation',
        )
        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observations_dim),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, actions_dim),
            name='actions',
        )
        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None,),
            name='rewards',
        )
        self._price_next_day_ph = tf.placeholder(
            tf.float32,
            shape=(None,),
            name='price_next_day',
        )
        return
    
    def _hidden_state_fun(self, state):
        with tf.variable_scope('hidden_state', reuse=tf.AUTO_REUSE):
            hidden_states = tf.layers.dense(state, self._hps.hidden_dim,
                                            activation=tf.nn.leaky_relu, name='state_hidden_layer')
        return hidden_states
    
    def _policy_fun(self, hidden_states):
        with tf.variable_scope('action_output', reuse=tf.AUTO_REUSE):
            actions = tf.nn.softmax(tf.layers.dense(hidden_states, self._actions_dim,
                                                    activation=tf.nn.sigmoid, name='action_output_layer'))
        return actions
    
    def _q_function(self, hidden_states, actions):
        with tf.variable_scope('Q_output', reuse=tf.AUTO_REUSE):
            q = tf.layers.dense(tf.concat([hidden_states, actions], axis=1), 1,
                                activation=None, name='Q_output_layer')
        return q
    
    def _next_policy_q(self):
        next_hidden_states = self._hidden_state_fun(self._next_observations_ph)
        next_actions = self._policy_fun(next_hidden_states)
        next_policy_q = self._q_function(next_hidden_states, next_actions)
        return next_policy_q
        
    def _build_graph(self):
        self._create_placeholders()
        hidden_states = self._hidden_state_fun(self._observations_ph)
        price_next_day = tf.layers.dense(hidden_states, 1, activation=None, name='price_output_layer')
        price_next_day = tf.squeeze(price_next_day)
        actions = self._policy_fun(hidden_states)
        policy_q = self._q_function(hidden_states, actions)
        
        actions_ph_q = self._q_function(hidden_states, self._actions_ph)
        next_policy_q = tf.stop_gradient(self._next_policy_q())
        
        # Calculate action loss and Q loss
        action_loss = -tf.reduce_sum(tf.squeeze(policy_q), axis=0, name='action_loss')
        
        q_loss = tf.reduce_sum(self._rewards_ph + tf.squeeze(self._hps.gamma * next_policy_q - actions_ph_q),
                               axis=0, name='q_loss')
        price_loss = tf.losses.mean_squared_error(self._price_next_day_ph, price_next_day)

        
        # Get update option
        t_vars = tf.trainable_variables()
        action_vars = [var for var in t_vars
                       if var.name.startswith('hidden_state') or var.name.startswith('action_output')]
        
        q_vars = [var for var in t_vars
                  if var.name.startswith('hidden_state') or var.name.startswith('Q_output')]
        
        action_train_opt = tf.train.AdamOptimizer(self._hps.learning_rate).minimize(
            action_loss, global_step=self.global_step, var_list=action_vars, name='action_train_opt')
        
        q_train_opt = tf.train.AdamOptimizer(self._hps.learning_rate).minimize(
            q_loss, global_step=self.global_step, var_list=q_vars, name='Q_train_opt')

        price_train_opt = tf.train.RMSPropOptimizer(self._hps.learning_rate).minimize(
            price_loss, global_step=self.global_step, name='price_train_opt')
        
        return actions, policy_q, price_next_day, action_loss, q_loss, price_loss, action_train_opt,\
               q_train_opt, price_train_opt


def main():
    hps = {'trunc_norm_init_std': 1e-4,
           'hidden_dim': 100,
           'train_dir': './model_test',
           'gamma': 0.99,
           'learning_rate': 0.001,
           'batch_size': 100,
           'encode_step': 200}
    hps = namedtuple("HParams", hps.keys())(**hps)
    
    if os.path.exists(hps.train_dir):
        shutil.rmtree(hps.train_dir, True)
        
    data_set = DataSet(hps)
    env = Env(hps, data_set)
    model = Model(hps, env.observations_dim, env.actions_dim)
    obs = env.reset()
    data_set.add_data(obs, 0, 0)

    start = time.time()
    data_size = 10000
    for i in range(data_size):
        print('\r{}/{}'.format(i, data_size), end='')
        obs, reward, _ = env.step(obs, Actions([0.3, 0.3, 0.4]))
        data_set.add_data(obs, 0, 0)

    print("[finished in {:.2f} s]".format(time.time() - start))
    n = 100
    for i in range(n):
        print('\n\n{}/{}'.format(i, n))
        model.price_train(1, data_set)
    return


if __name__ == '__main__':
    np.set_printoptions(2)
    main()

