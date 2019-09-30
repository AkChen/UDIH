import tensorflow as tf
import numpy as np

'''
 Author:AkChen 
 Mail:  18120348@bjtu.edu.cn
'''


class RAE:
    def __init__(self, sess, num_of_layers=None, num_per_layer=None, active_function_per_layer=None, learning_rate=None,
                 RAE_id=None,
                 input=None, beta=0.001, loss_type='l2', recover_config=None):  # fc_ae
        self.config = {}

        if not recover_config is None:
            self.config['num_of_layers'] = num_of_layers = recover_config['num_of_layers']
            self.config['num_per_layer'] = num_per_layer = recover_config['num_per_layer']
            self.config['id'] = RAE_id = recover_config['id']
            self.config['active'] = active_function_per_layer = recover_config['active']
            self.config['learning_rate'] = learning_rate = recover_config['learning_rate']
            self.config['loss_type'] = loss_type = recover_config['loss_type']

        self.config['num_of_layers'] = num_of_layers
        self.config['num_per_layer'] = num_per_layer
        self.config['layers'] = [{} for l in range(num_of_layers)]
        self.config['input'] = input
        self.config['id'] = RAE_id
        self.config['active'] = active_function_per_layer
        self.config['learning_rate'] = learning_rate
        self.config['loss_type'] = loss_type

        # the first layer's input
        if input is None:
            self.config['layers'][0]['input'] = tf.placeholder(tf.float32, (None, num_per_layer[0]),
                                                               name='RAE_' + str(RAE_id) + '_layer_0_input')
            self.config['layers'][0]['output'] = self.config['layers'][0]['input']
            self.config['layers'][0]['num_of_node'] = num_per_layer[0]
            self.config['input'] = self.config['layers'][0]['input']
        else:
            self.config['layers'][0]['input'] = input
            self.config['layers'][0]['output'] = input
            self.config['layers'][0]['num_of_node'] = num_per_layer[0]
            self.config['input'] = input

        # the last layer's output is residual
        for l in range(1, num_of_layers):
            if recover_config is None:
                self.config['layers'][l]['W'] = tf.Variable(
                    tf.random_uniform([num_per_layer[l - 1], num_per_layer[l]], -1.0, 1.0),
                    name='RAE_' + str(RAE_id) + '_layer_' + str(l) + '_W')
            else:
                self.config['layers'][l]['W'] = tf.Variable(recover_config['layers'][l - 1]['W'],
                                                            name='RAE_' + str(RAE_id) + '_layer_' + str(l) + '_W')

            if recover_config is None:
                self.config['layers'][l]['b'] = tf.Variable(tf.random_uniform([num_per_layer[l]], -1.0, 1.0),
                                                            name='RAE_' + str(RAE_id) + '_layer_' + str(l) + '_b')
            else:
                self.config['layers'][l]['b'] = tf.Variable(recover_config['layers'][l - 1]['b'],
                                                            name='RAE_' + str(RAE_id) + '_layer_' + str(l) + '_b')

            self.config['layers'][l]['input'] = tf.nn.xw_plus_b(self.config['layers'][l - 1]['output'],
                                                                self.config['layers'][l]['W'],
                                                                self.config['layers'][l]['b'],
                                                                name='RAE_' + str(RAE_id) + '_layer_' + str(
                                                                    l) + '_input')

            # output

            if active_function_per_layer[l] == 'sigmoid':
                self.config['layers'][l]['output'] = tf.nn.sigmoid(self.config['layers'][l]['input'],
                                                                   name='RAE_' + str(RAE_id) + '_layer_' + str(
                                                                       l) + '_output')
            if active_function_per_layer[l] == 'relu':
                self.config['layers'][l]['output'] = tf.nn.relu(self.config['layers'][l]['input'],
                                                                name='RAE_' + str(RAE_id) + '_layer_' + str(
                                                                    l) + '_output')
            if active_function_per_layer[l] == 'tanh':
                self.config['layers'][l]['output'] = tf.nn.tanh(self.config['layers'][l]['input'],
                                                                name='RAE_' + str(RAE_id) + '_layer_' + str(
                                                                    l) + '_output')

            # drop out
            # self.config['layers'][l]['output'] = tf.nn.dropout(self.config['layers'][l]['output'],0.9)

            self.config['layers'][l]['num_of_node'] = num_per_layer[l]

        self.config['residual'] = self.config['layers'][-1]['output']
        self.config['output'] = tf.add(self.config['input'], self.config['residual'],
                                       name=str(RAE_id) + '_full_output')
        # desire redidual
        self.config['d_residual'] = tf.placeholder(tf.float32, (None, self.config['num_per_layer'][0]),
                                                   name='RAE_' + str(RAE_id) + '_d_residual')
        weight_decay = tf.nn.l2_loss(self.config['layers'][1]['W']) + tf.nn.l2_loss(self.config['layers'][1]['b'])
        for d in range(2, self.config['num_of_layers']):
            weight_decay += tf.nn.l2_loss(self.config['layers'][l]['W']) + tf.nn.l2_loss(self.config['layers'][l]['b'])
        if self.config['loss_type'] == 'l2':
            self.config['loss'] = tf.nn.l2_loss(tf.subtract(self.config['residual'], self.config['d_residual']),
                                                name='RAE_' + str(RAE_id) + '_l2_loss') + weight_decay * beta

        if self.config['loss_type'] == 'log':
            self.config['loss'] = tf.nn.softmax_cross_entropy_with_logits(logits=self.config['residual'],
                                                                              labels=self.config['d_residual'])
        if self.config['loss_type'] == 'l1':
            self.config['loss'] = tf.abs(self.config['residual']-self.config['d_residual'])

        global_step = tf.Variable(0, trainable=False)
        lr_step = tf.train.exponential_decay(learning_rate, global_step, 5000, 0.95, staircase=True)
        # self.config['opt'] = tf.train.MomentumOptimizer(lr_step,1.0)

        self.config['opt'] = tf.train.AdamOptimizer(lr_step) #tf.train.GradientDescentOptimizer(lr_step)##
        self.config['update'] = self.config['opt'].minimize(self.config['loss'])

        init = tf.global_variables_initializer()
        sess.run(init)

    def train(self, sess, d_residual, epoch, batch_size=200, input=None,
              show_loss_epoch=0,test_last_full =None,test_full = None):  # input is none if and only if it is not the first block.

        for e in range(epoch):
            # batch it
            # shuffle
            cur_index = 0
            length = d_residual.shape[0]
            while cur_index < length:
                if cur_index + batch_size < length:
                    cur_d_res = d_residual[cur_index:cur_index + batch_size]
                    cur_input = input[cur_index:cur_index + batch_size]
                else:
                    cur_d_res = d_residual[cur_index:length]
                    cur_input = input[cur_index:length]
                cur_index += batch_size

                _ = sess.run(self.config['update'], feed_dict={self.config['layers'][0]['input']: cur_input,
                                                               self.config['d_residual']: cur_d_res})
            if show_loss_epoch > 0 and e % show_loss_epoch == 0:
                predict_res = sess.run(self.config['residual'], feed_dict={self.config['layers'][0]['input']: input})
                RA_res_loss = np.sum(np.square(d_residual-predict_res))
                print('RAE_' + str(self.config['id']) + '_epoch:%d loss:%f' % (e, RA_res_loss))
                if not (test_last_full is None):
                    test_predict_full = self.predict_full(sess,input=test_last_full)
                    test_loss = np.sum(np.square(test_predict_full-test_full))
                    print('RAE_' + str(self.config['id']) + '_epoch:%d test_loss:%f' % (e, test_loss ))

    def predict_full(self, sess, input=None):  # input is none if and only if it is not the first block.
        if input is None:
            self.config['output_np'] = np.asarray(sess.run(self.config['output']))
            return self.config['output_np']
        else:
            self.config['output_np'] = np.asarray(
                sess.run(self.config['output'], feed_dict={self.config['layers'][0]['input']: input}))
            return self.config['output_np']

    def predict_res(self, sess, input=None):
        if input is None:
            self.config['residual_np'] = np.asarray(sess.run(self.config['residual']))
            return self.config['residual_np']
        else:
            self.config['residual_np'] = np.asarray(
                sess.run(self.config['residual'], feed_dict={self.config['layers'][0]['input']: input}))
            return self.config['residual_np']

    def get_recover_config(self, sess):
        config = {}
        config['num_of_layers'] = self.config['num_of_layers']
        config['num_per_layer'] = self.config['num_per_layer']
        config['active'] = self.config['active']
        config['id'] = self.config['id']
        config['layers'] = []
        config['learning_rate'] = self.config['learning_rate']
        config['loss_type'] = self.config['loss_type']
        for idx in range(1, len(self.config['layers'])):
            layer_config = {}
            layer = self.config['layers'][idx]
            layer_config['W'] = np.asarray(sess.run(layer['W']))
            layer_config['b'] = np.asarray(sess.run(layer['b']))
            config['layers'].append(layer_config)
        return config
