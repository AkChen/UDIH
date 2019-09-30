import tensorflow as tf
import numpy as np


class Discriminator:
    def __init__(self, sess, num_of_layers, num_per_layer, learning_rate, recover_config=None,weight_decay = 0.001):  # default sigmoid
        self.config = {}
        if recover_config is None:
            self.config['layers'] = [{} for i in range(num_of_layers)]
            self.config['num_of_layer'] = num_of_layers
            self.config['num_per_layer'] = num_per_layer

            self.config['layers'][0]['input'] = tf.placeholder(tf.float32, (None, num_per_layer[0]), name='DIS_0_input')
            self.config['layers'][0]['output'] = self.config['layers'][0]['input']
            for i in range(1, len(num_per_layer)):
                self.config['layers'][i] = {}
                self.config['layers'][i]['W'] = tf.Variable(
                    tf.truncated_normal([num_per_layer[i - 1], num_per_layer[i]], 0.0, 0.1),
                    name='DIS_layer_' + str(i) + '_W')
                self.config['layers'][i]['b'] = tf.Variable(tf.constant(0.0,[num_per_layer[i]]),
                                                            name='DIS_layer_' + str(i) + '_b')
                self.config['layers'][i]['input'] = tf.nn.xw_plus_b(self.config['layers'][i - 1]['output'],
                                                                    self.config['layers'][i]['W'],
                                                                    self.config['layers'][i]['b'])
                self.config['layers'][i]['output'] = tf.nn.sigmoid(self.config['layers'][i]['input'])

            self.config['desire_output'] = tf.placeholder(tf.float32, [None, num_per_layer[-1]],
                                                          name='loss_input')  # 计算对输入的梯度

            global_step2 = tf.Variable(0, trainable=False)
            lr_step2 = tf.train.exponential_decay(learning_rate*100, global_step2, 2000, 0.98, staircase=True)
            self.config['input_gradient'] = tf.subtract(self.config['layers'][-1]['output'],
                                                        self.config['desire_output']) * lr_step2

            for i in range(num_of_layers - 1):
                cur_layer = self.config['layers'][num_of_layers - i - 1]
                self.config['input_gradient'] = tf.matmul(tf.multiply(self.config['input_gradient'],
                                                                      tf.multiply(cur_layer['output'],
                                                                                  1 - cur_layer['output'])),
                                                          tf.transpose(cur_layer['W']))

            self.config['loss'] = tf.nn.l2_loss(
                tf.subtract(self.config['layers'][-1]['output'], self.config['desire_output']))
            for i in range(1, len(num_per_layer)):
                self.config['loss'] += weight_decay * (tf.nn.l2_loss(self.config['layers'][i]['W']) + tf.nn.l2_loss(self.config['layers'][i]['b']))
            global_step = tf.Variable(0, trainable=False)
            lr_step = tf.train.exponential_decay(learning_rate, global_step, 2000, 0.98, staircase=True)
            # self.config['opt'] = tf.train.MomentumOptimizer(lr_step,1.0)
            self.config['opt'] = tf.train.GradientDescentOptimizer(lr_step)
            self.config['update'] = self.config['opt'].minimize(self.config['loss'])
            init = tf.global_variables_initializer()
            sess.run(init)

    def get_accuracy(self, sess, input, label):
        output = sess.run(self.config['layers'][-1]['output'], feed_dict={self.config['layers'][0]['input']: input})
        result_label = []
        num_succ = 0
        for i, o in enumerate(output):
            if o[0] > o[1]:
                if (label[i][0] > label[i][1]):
                    num_succ += 1
            else:
                if (label[i][0] < label[i][1]):
                    num_succ += 1
        return num_succ / len(input)
