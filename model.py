import tensorflow as tf
from env import Env

class Model:
    def __init__(self, env, hps):
        self._env = env
        self._hps = hps
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.action, self.Q, self.action_loss, self.Q_loss, self.action_train_opt, self.Q_train_opt = \
            self._build_graph()
        self._sess, self._summary_writer = self._sess_setup()
        return
    
    def train(self, iteration, data_set):
        for i in range(iteration):
            obs, next_obs, reward = data_set.get_batch(self._hps.batch_size)
            action_loss, Q_loss = self._train_one_step(obs, next_obs, reward)
            print('action_loss: {}, Q_loss : {}'.format(action_loss, Q_loss))
        
        return
    
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
        observations_dim = self._env.observations_dim
        actions_dim = self._env.actions_dim
        
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
        #         self._actions_ph = tf.placeholder(
        #             tf.float32,
        #             shape=(None, actions_dim),
        #             name='actions',
        #         )
        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None,),
            name='rewards',
        )
        return
    
    #     def _linear(self, arg, output_size, activation, scope=None, reuse=False):
    #         input_size = arg.get_shape().as_list()[1]
    #         print('input_size', input_size)
    #         trunc_norm_init = tf.truncated_normal_initializer(stddev=self._hps.trunc_norm_init_std)
    
    #         with tf.variable_scope(scope or "Linear", reuse=reuse):
    #             matrix = tf.get_variable("Matrix", [input_size, output_size])
    #             res = tf.matmul(arg, matrix)
    #             bias_term = tf.get_variable("Bias", [output_size],
    #                                         initializer=trunc_norm_init)
    #         return activation(res + bias_term)
    
    def _action_Q_output(self, state):
        with tf.variable_scope('hidden_state', reuse=tf.AUTO_REUSE):
            hidden_states = tf.layers.dense(state, self._hps.hidden_dim,
                                            activation=tf.nn.sigmoid, name='state_hidden_layer')
        
        with tf.variable_scope('action_output', reuse=tf.AUTO_REUSE):
            actions = tf.nn.softmax(tf.layers.dense(hidden_states, self._env.actions_dim,
                                                    activation=tf.nn.sigmoid, name='action_output_layer'))
        
        with tf.variable_scope('Q_output', reuse=tf.AUTO_REUSE):
            Q = tf.layers.dense(tf.concat([hidden_states, actions], axis=1), 1,
                                activation=None, name='Q_output_layer')
        return hidden_states, actions, Q
    
    def _build_graph(self):
        self._create_placeholders()
        _, action, Q = self._action_Q_output(self._observations_ph)
        _, _, next_Q = tf.stop_gradient(self._action_Q_output(self._next_observations_ph))
        
        # Calculate action loss and Q loss
        action_loss = -tf.reduce_sum(tf.squeeze(Q), axis=0)
        
        Q_loss = tf.reduce_sum(self._rewards_ph + tf.squeeze(self._hps.gamma * next_Q - Q), axis=0)
        
        # Get update option
        t_vars = tf.trainable_variables()
        action_vars = [var for var in t_vars
                       if var.name.startswith('hidden_state') or var.name.startswith('action_output')]
        
        Q_vars = [var for var in t_vars
                  if var.name.startswith('hidden_state') or var.name.startswith('Q_output')]
        
        action_train_opt = tf.train.AdamOptimizer(self._hps.learning_rate).minimize(
            action_loss, var_list=action_vars)
        
        Q_train_opt = tf.train.AdamOptimizer(self._hps.learning_rate).minimize(
            action_loss, var_list=Q_vars)
        
        return action, Q, action_loss, Q_loss, action_train_opt, Q_train_opt


def main():
    hps = {'trunc_norm_init_std': 1e-4,
           'hidden_dim': 20,
           'train_dir': './model',
           'gamma': 0.99,
           'learning_rate': 0.003,
           'batch_size': 10,
           'days': 20}
    env = Env()
    model = Model(env, hps)
    
    return


if __name__ == '__main__':
    main()
