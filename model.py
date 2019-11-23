import tensorflow as tf
from env import *
from collections import namedtuple
import os
import shutil
from data_set import DataSet
import time
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import json


def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]
  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return res + bias_term


def variable_summaries(var_name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries_{}'.format(var_name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        
        
class Model:
    def __init__(self, hps, obs_dim, actions_dim):
        self._hps = hps
        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.rand_unif_init = tf.random_uniform_initializer(-0.1, 0.1, seed=123)
        self.trunc_norm_init = tf.truncated_normal_initializer(stddev=0.1)
        
        # self.action, self.Q, self.price_predict, self.action_loss, self.Q_loss, self.price_loss, \
        #     self.action_train_opt, self.Q_train_opt, self.price_train_opt = self._build_graph()

        self.price_predict, self.price_loss, self.accuracy, self.price_train_opt = self._build_graph()
        self.summaries = tf.summary.merge_all()
        self._sess, self._summary_writer = self._sess_setup()
        return
    
    def policy_q_train(self, iteration, data_set):
        for i in range(iteration):
            obs, next_obs, reward = data_set.get_batch(self._hps.batch_size)
            action_loss, Q_loss = self._train_one_step(obs, next_obs, reward)
            print('action_loss: {}, Q_loss : {}'.format(action_loss, Q_loss))
        
        return

    def price_print(self, price, price_ph, price_loss, tf_accuracy, prefix, step):
        predict_label = [[x[0], -0.3 < x[0] - x[1] < 0.3] for x in list(zip(price[:, 0], price_ph[:, 0]))
                         if x[0] > 0.7 or x[0] < 0.3]
        # print('price_ph: ', price_ph)
        # print('price: ', price)
        # print('predict_label: ', predict_label)
        print(prefix+'loss: ', price_loss)
        predict_label = np.array(predict_label)
        high_confident_rate = 100 * len(predict_label) / len(price)
        high_confident_acc = 100 * np.sum(predict_label[:, 1]) / len(predict_label) \
            if len(predict_label) != 0 else 50
        print(prefix+'High confident predict rate: {:.2f}%'.format(high_confident_rate))
        print(prefix+'High confident predict accuracy: {:.2f}%'.format(high_confident_acc))
        print(prefix+'Overall accuracy: {}'.format(tf_accuracy))

        summary = tf.Summary()
        summary.value.add(tag=prefix+'loss', simple_value=price_loss)
        summary.value.add(tag=prefix+'high_confident_rate', simple_value=high_confident_rate)
        summary.value.add(tag=prefix+'high_confident_acc', simple_value=high_confident_acc)
        summary.value.add(tag=prefix+'overall_acc', simple_value=tf_accuracy)
        self._summary_writer.add_summary(summary, step)
        
    def price_test(self, iteration, data_set):
        for i in range(iteration):
            obs_ph, price_ph = data_set.get_price_test_batch(self._hps.batch_size)
            # obs_ph = (obs_ph - 50) / 100
            # print('obs_ph\n', obs_ph)
            to_return = [self.price_predict, self.price_loss, self.accuracy, self.global_step]
            feed_dict = {self._observations_ph: obs_ph, self._price_up_down_prob_ph: price_ph,
                         self._keep_prob_ph: 1}
            price, price_loss, tf_accuracy, train_step = self._sess.run(to_return, feed_dict)
            self.price_print(price, price_ph, price_loss, tf_accuracy, 'test_', train_step)
            
    def price_train(self, iteration, data_set):
        for i in range(iteration):
            obs_ph, price_ph = data_set.get_price_batch(self._hps.batch_size)
            keep_prob_ph = self._hps.keep_prob
            # obs_ph = (obs_ph - 50) / 100
            # print('obs_ph\n', obs_ph)
            to_return = [self.price_predict, self.price_loss, self.price_train_opt, self.summaries, self.global_step,
                         self.accuracy]
            feed_dict = {self._observations_ph: obs_ph, self._price_up_down_prob_ph: price_ph,
                         self._keep_prob_ph: keep_prob_ph}
            price, price_loss, _, summaries, train_step, tf_accuracy = self._sess.run(to_return, feed_dict)
            
            self._summary_writer.add_summary(summaries, train_step)
            self.price_print(price, price_ph, price_loss, tf_accuracy, 'train_', train_step)
            
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
                                 save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
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

        self._keep_prob_ph = tf.placeholder(
            tf.float32,
            name='keep_prob',
        )
        
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
        self._price_up_down_prob_ph = tf.placeholder(
            tf.float32,
            shape=(None, 2),
            name='up_down_prob',
        )
        return

    def _add_encoder(self, emb_enc_inputs, seq_len):
        """Add a single-layer bidirectional LSTM encoder to the graph.
  
        Args:
          emb_enc_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
          seq_len: Lengths of emb_enc_inputs (before padding). A tensor of shape [batch_size].
  
        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
          fw_state, bw_state:
            Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.enc_hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.enc_hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, emb_enc_inputs,
                                                                                dtype=tf.float32,
                                                                                swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states
        return encoder_outputs, fw_st, bw_st

    def _add_decoder(self, _enc_states, emb_dec_inputs, _dec_in_state):
        with tf.variable_scope('decoder'):
            cell = tf.contrib.rnn.LSTMCell(self._hps.dec_hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
            # if this line fails, it's because the attention length isn't defined
            attn_size = _enc_states.get_shape()[2]
            emb_size = emb_dec_inputs[0].get_shape()[0]  # if this line fails, it's because the embedding isn't defined
            decoder_attn_size = _dec_in_state.c.get_shape()[1]
            tf.logging.info("attn_size: %i, emb_size: %i", attn_size, emb_size)
            attention_vec_size = attn_size
            _enc_states = tf.expand_dims(_enc_states, axis=2)  # now is shape (batch_size, max_enc_steps, 1, attn_size)
            emb_dim = emb_dec_inputs.get_shape().with_rank(2)[1]
            if emb_dim is None:
                raise ValueError("Could not infer input size from input: %s" % emb_dec_inputs.name)

            W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
            v = variable_scope.get_variable("v", [attention_vec_size])
            encoder_features = nn_ops.conv2d(_enc_states, W_h, [1, 1, 1, 1],
                                             "SAME")  # shape (batch_size, max_enc_steps, 1, attention_vec_size)

            def attention(decoder_state):
                with variable_scope.variable_scope("Attention"):
                    decoder_features = linear(decoder_state, attention_vec_size,
                                              True)  # shape (batch_size, attention_vec_size)
                    decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1),
                                                      1)  # reshape to (batch_size, 1, 1, attention_vec_size)
        
                    # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                    e_not_masked = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features),
                                                       [2, 3])  # calculate e, (batch_size, max_enc_steps)
                    masked_e = nn_ops.softmax(e_not_masked)  # (batch_size, max_enc_steps)
                    masked_sums = tf.reduce_sum(masked_e, axis=1)  # shape (batch_size)
                    masked_e = masked_e / tf.reshape(masked_sums, [-1, 1])
                    attn_dist = masked_e
                    context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [-1, self._hps.encode_step, 1, 1])
                                                         * _enc_states, [1, 2])  # shape (batch_size, attn_size).
                    context_vector = array_ops.reshape(context_vector, [-1, attn_size])
        
                    return context_vector

            context_vector = attention(_dec_in_state)
            x = linear([emb_dec_inputs] + [context_vector], emb_dim, True)
            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, _dec_in_state)
            context_vector = attention(state)
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + [context_vector], cell.output_size, True)
            
            cell_output, state = cell(emb_dec_inputs, _dec_in_state)
            return cell_output
    
    def _attention(self, enc_output):
        # enc_output: [batch_size, enc_steps, 2*hidden_dim]

        with variable_scope.variable_scope("Attention"):
            _enc_states = enc_output[:, :-1, :]     # [batch_size, enc_steps-1, 2*hidden_dim]
            _dec_states = enc_output[:, -1, :]  # [batch_size, 2*hidden_dim]
            attn_size = _enc_states.get_shape()[2]       # 2*hidden_dim
            W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attn_size])
            v = variable_scope.get_variable("v", [attn_size])
            _enc_states = tf.expand_dims(_enc_states, axis=2)  # now is shape (batch_size, enc_steps-1, 1, attn_size)
            encoder_features = nn_ops.conv2d(_enc_states, W_h, [1, 1, 1, 1],
                                             "SAME")  # shape (batch_size, enc_steps-1, 1, attn_size)

            decoder_features = linear(_dec_states, attn_size, True, scope='decoder_features')  # shape (batch_size, attn_size)
            decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1),
                                              1)  # reshape to (batch_size, 1, 1, attn_size)
            
            e_not_masked = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features),
                                               [2, 3])  # calculate e, (batch_size, enc_steps-1)
            attn_dist = nn_ops.softmax(e_not_masked)  # (batch_size, enc_steps-1)
            context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [-1, self._hps.encode_step-1, 1, 1])
                                                 * _enc_states, [1, 2])  # shape (batch_size, attn_size).
            context_vector = array_ops.reshape(context_vector, [-1, attn_size])
            print('_dec_states: ', _dec_states)
            print('context_vector: ', context_vector)
            output = linear([_dec_states] + [context_vector], attn_size, True, scope='output')
            return output
    
    def _hidden_state_fun(self, obs_state):
        with tf.variable_scope('hidden_state', reuse=tf.AUTO_REUSE):  # tf.AUTO_REUSE
            obs = obs_state[:, 0: self._hps.encode_dim * self._hps.encode_step]
            # state = obs_state[:, self._hps.encode_dim * self._hps.encode_step:]  # 当前的持有状态信息
            obs = tf.reshape(obs, [-1, self._hps.encode_step, self._hps.encode_dim])
            enc_output, fw_st, bw_st = self._add_encoder(obs, self._hps.encode_step)
            output = self._attention(enc_output)
            # dec_in_state = self._reduce_states(fw_st, bw_st)
            # hidden_state = self._add_decoder(enc_output, state, dec_in_state)
        
        return output
    
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
        hidden_states = tf.nn.dropout(hidden_states, keep_prob=self._keep_prob_ph)
        price_logits = tf.layers.dense(hidden_states, 2, activation=None, name='price_prob_output')
        price_up_down_prob = tf.nn.softmax(price_logits)
        variable_summaries('predict', price_up_down_prob)

        # actions = self._policy_fun(hidden_states)
        # policy_q = self._q_function(hidden_states, actions)
        #
        # actions_ph_q = self._q_function(hidden_states, self._actions_ph)
        # next_policy_q = tf.stop_gradient(self._next_policy_q())
        #
        # # Calculate action loss and Q loss
        # action_loss = -tf.reduce_sum(tf.squeeze(policy_q), axis=0, name='action_loss')
        #
        # q_loss = tf.reduce_sum(self._rewards_ph + tf.squeeze(self._hps.gamma * next_policy_q - actions_ph_q),
        #                        axis=0, name='q_loss')

        correct_prediction = tf.cast(tf.equal(tf.argmax(price_up_down_prob, 1),
                                              tf.argmax(self._price_up_down_prob_ph, 1)), tf.float32)
        variable_summaries('accuracy', correct_prediction)
        accuracy = tf.reduce_mean(correct_prediction)
        price_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._price_up_down_prob_ph,
                                                                logits=price_logits)
        variable_summaries('price_loss', price_loss)
        price_loss = tf.reduce_mean(price_loss)
        
        # price_loss = tf.losses.mean_squared_error(self._price_next_day_ph, price_next_day)
        #
        # # Get update option
        # t_vars = tf.trainable_variables()
        # action_vars = [var for var in t_vars
        #                if var.name.startswith('hidden_state') or var.name.startswith('action_output')]
        #
        # q_vars = [var for var in t_vars
        #           if var.name.startswith('hidden_state') or var.name.startswith('Q_output')]
        #
        # action_train_opt = tf.train.AdamOptimizer(self._hps.learning_rate).minimize(
        #     action_loss, global_step=self.global_step, var_list=action_vars, name='action_train_opt')
        #
        # q_train_opt = tf.train.AdamOptimizer(self._hps.learning_rate).minimize(
        #     q_loss, global_step=self.global_step, var_list=q_vars, name='Q_train_opt')

        price_train_opt = tf.train.AdamOptimizer(self._hps.learning_rate).minimize(
            price_loss, global_step=self.global_step, name='price_train_opt')
        
        # return actions, policy_q, price_up_down_prob, action_loss, q_loss, price_loss, action_train_opt,\
        #        q_train_opt, price_train_opt
        return price_up_down_prob, price_loss, accuracy, price_train_opt
        
    def _reduce_states(self, fw_st, bw_st):
        """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.
  
        Args:
          fw_st: LSTMStateTuple with hidden_dim units.
          bw_st: LSTMStateTuple with hidden_dim units.
  
        Returns:
          state: LSTMStateTuple with hidden_dim units.
        """
        enc_hidden_dim = self._hps.enc_hidden_dim
        dec_hidden_dim = self._hps.dec_hidden_dim
    
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [enc_hidden_dim * 2, dec_hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [enc_hidden_dim * 2, dec_hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [dec_hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [dec_hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
        
            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state
    
    
def write_info(info, file_name):
    with open(file_name, "w") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
        f.write('\n')


def get_hps():
    hps = {'enc_hidden_dim': 10,
           'dec_hidden_dim': 100,
           'gamma': 0.99,
           'learning_rate': 0.0003,
           'batch_size': 256,
           'encode_step': 60,  # 历史数据个数
           'keep_prob': 0.5,
           'encode_dim': 1,   # 特征个数：时间，开，收，高，低，量
           
           'train_data_num': 100000,  # 训练集个数
           'train_iter': 50000,    # 训练的 iterations
           'eval_interval': 20,  # 每次测试间隔的训练次数
           
           'exp_name': '60相对收盘价注意力10h_dim_0.5d',   # 实验名称
           'model_dir': './model',      # 保存模型文件夹路径
           'is_retrain': False}      # 是否从头训练
    hps['train_dir'] = os.path.join(hps['model_dir'], hps['exp_name'])

    if os.path.exists(hps['train_dir']):
        if hps['is_retrain']:
            shutil.rmtree(hps['train_dir'], True)
    else:
        os.makedirs(hps['train_dir'])
        
    write_info(hps, os.path.join(hps['train_dir'], 'parameters.txt'))
    hps = namedtuple("HParams", hps.keys())(**hps)
    return hps


def main():
    hps = get_hps()
    data_set = DataSet(hps)
    env = Env(hps, data_set)
    model = Model(hps, env.observations_dim, env.actions_dim)
    obs = env.reset()
    data_set.add_data(obs, 0, 0)

    data_size = hps.train_data_num
    for i in range(data_size):
        print('\r{}/{}'.format(i, data_size), end='')
        obs, reward, _ = env.step(obs, Actions([0.3, 0.3, 0.4]))
        data_set.add_data(obs, 0, 0)

    n = hps.train_iter
    for i in range(n):
        print('\n\n{}/{}'.format(i, n))
        model.price_train(1, data_set)
        if i % hps.eval_interval == 0:
            print('-'*50)
            model.price_test(1, data_set)
            print('-'*50)
        
    return


if __name__ == '__main__':
    start = time.time()
    np.set_printoptions(2)
    main()
    print("[finished in {:.2f} s]".format(time.time() - start))

