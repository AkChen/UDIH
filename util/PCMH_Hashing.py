import tensorflow as tf
import numpy as np


class PCMH:
    def __init__(self, image_dim, image_hidden_dim, text_dim, text_hidden_dim, bits, learning_rate, delta=2.0,
                 weight_decay=0.1,
                 recover_config=None):
        if (not (recover_config is None)):
            self.config = {}
            self.config['learning_rate'] = recover_config['learning_rate']

        else:  # init
            self.config = {}
            self.config['learning_rate'] = learning_rate
            self.config['image_dim'] = image_dim
            self.config['image_hidden_dim'] = image_hidden_dim
            self.config['text_dim'] = text_dim
            self.config['text_hidden_dim'] = text_hidden_dim
            self.config['bits'] = bits
            self.config['delta'] = delta
            self.config['weight_decay'] = weight_decay

        #

        # input
        self.config['image_rel_input'] = tf.placeholder(tf.float32, [None, self.config['image_dim']],
                                                        name='image_rel_input')
        self.config['image_irrel_input'] = tf.placeholder(tf.float32, [None, self.config['image_dim']],
                                                          name='image_irrel_input')
        self.config['text_rel_input'] = tf.placeholder(tf.float32, [None, self.config['text_dim']],
                                                       name='text_rel_input')
        self.config['text_irrel_input'] = tf.placeholder(tf.float32, [None, self.config['text_dim']],
                                                         name='text_irrel_input')
        self.config['rel_mul'] = tf.placeholder(tf.float32, [None], name='rel_mul')
        self.config['irrel_mul'] = tf.placeholder(tf.float32, [None], name='irrel_mul')

        # weights and bias
        if (recover_config is None):
            self.config['IW1'] = tf.Variable(
                tf.truncated_normal([self.config['image_dim'], self.config['image_hidden_dim']], mean=0., stddev=0.1),
                name='IW1')
            self.config['Ib1'] = tf.Variable(
                tf.constant(0.0, tf.float32, shape=[self.config['image_hidden_dim']]),
                name='Ib1')
            self.config['IW2'] = tf.Variable(
                tf.truncated_normal([self.config['image_hidden_dim'], self.config['bits']], mean=0., stddev=0.1),
                name='IW2')
            self.config['Ib2'] = tf.Variable(tf.constant(0.0, tf.float32, shape=[self.config['bits']]),
                                             name='Ib2')

            self.config['TW1'] = tf.Variable(
                tf.truncated_normal([self.config['text_dim'], self.config['text_hidden_dim']], mean=0., stddev=0.1),
                name='TW1')
            self.config['Tb1'] = tf.Variable(
                tf.constant(0.0, tf.float32, shape=[self.config['text_hidden_dim']]),
                name='Tb1')
            self.config['TW2'] = tf.Variable(
                tf.truncated_normal([self.config['text_hidden_dim'], self.config['bits']], mean=0., stddev=0.1),
                name='TW2')
            self.config['Tb2'] = tf.Variable(tf.constant(0.0, tf.float32, shape=[self.config['bits']]),
                                             name='Ib2')

        self.config['image_rel_rep'] = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.config['image_rel_input'], self.config['IW1'], self.config['Ib1'])),
            self.config['IW2'], self.config['Ib2'])
        self.config['image_irrel_rep'] = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.config['image_irrel_input'], self.config['IW1'], self.config['Ib1'])),
            self.config['IW2'], self.config['Ib2'])
        self.config['text_rel_rep'] = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.config['text_rel_input'], self.config['TW1'], self.config['Tb1'])),
            self.config['TW2'], self.config['Tb2'])
        self.config['text_irrel_rep'] = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.config['text_irrel_input'], self.config['TW1'], self.config['Tb1'])),
            self.config['TW2'], self.config['Tb2'])

        img_mid_rel = tf.nn.tanh(
            tf.nn.xw_plus_b(self.config['image_rel_input'], self.config['IW1'], self.config['Ib1']))
        img_mid_irrel = tf.nn.tanh(
            tf.nn.xw_plus_b(self.config['image_irrel_input'], self.config['IW1'], self.config['Ib1']))
        txt_mid_rel = tf.nn.tanh(tf.nn.xw_plus_b(self.config['text_rel_input'], self.config['TW1'], self.config['Tb1']))
        txt_mid_irrel = tf.nn.tanh(
            tf.nn.xw_plus_b(self.config['text_irrel_input'], self.config['TW1'], self.config['Tb1']))

        self.config['image_rel_sig'] = tf.sigmoid(self.config['image_rel_rep'])
        self.config['image_irrel_sig'] = tf.sigmoid(self.config['image_irrel_rep'])
        self.config['text_rel_sig'] = tf.sigmoid(self.config['text_rel_rep'])
        self.config['text_irrel_sig'] = tf.sigmoid(self.config['text_irrel_rep'])

        self.config['image_rel_hash'] = tf.sign(
            self.config['image_rel_sig'] - 0.5)  # tf.cast(tf.add(self.config['image_rel_sig'], 0.5), tf.int32)
        self.config['text_rel_hash'] = tf.sign(
            self.config['text_rel_sig'] - 0.5)  # tf.cast(tf.add(self.config['text_rel_sig'], 0.5), tf.int32)
        # self.config['image_irrel_hash'] = tf.sign(self.config['image_irrel_sig'] - 0.5)
        # self.config['text_irrel_hash'] = tf.sign(self.config['text_rel_sig'] - 0.5)

        self.config['rel_distance'] = tf.multiply(tf.reduce_sum(
            tf.square(tf.subtract(self.config['image_rel_sig'], self.config['text_rel_sig'])), 1),
            self.config['rel_mul']) + 0*tf.multiply(tf.reduce_sum(tf.square(tf.subtract(img_mid_rel, txt_mid_rel)), 1),
                                                  self.config['rel_mul'])

        self.config['i2t_irrel_distance'] = tf.multiply(tf.reduce_sum(
            tf.square(tf.subtract(self.config['image_rel_sig'], self.config['text_irrel_sig'])), 1),
            self.config['irrel_mul']) + 0*tf.multiply(tf.reduce_sum(
            tf.square(tf.subtract(img_mid_rel, txt_mid_irrel)), 1),
            self.config['irrel_mul'])
        self.config['t2i_irrel_distance'] = tf.multiply(tf.reduce_sum(tf.square(tf.subtract(self.config['text_rel_sig'],
                                                                                            self.config[
                                                                                                'image_irrel_sig'])),
                                                                      1), self.config['irrel_mul']) +0*tf.multiply(tf.reduce_sum(tf.square(tf.subtract(txt_mid_rel,img_mid_irrel)),
                                                                      1), self.config['irrel_mul'])

        weight_decay_sum = self.config['weight_decay'] * (tf.nn.l2_loss(self.config['IW1']) + \
                                                          tf.nn.l2_loss(self.config['IW2']) + \
                                                          tf.nn.l2_loss(self.config['TW1']) + \
                                                          tf.nn.l2_loss(self.config['TW2']) + \
                                                          tf.nn.l2_loss(self.config['Ib1']) + \
                                                          tf.nn.l2_loss(self.config['Ib2']) + \
                                                          tf.nn.l2_loss(self.config['Tb1']) + \
                                                          tf.nn.l2_loss(self.config['Tb2']))

        self.config['i2t_sim_loss'] = tf.reduce_mean(
            tf.maximum(0.0, self.config['delta'] + self.config['rel_distance'] - self.config[
                'i2t_irrel_distance'])) + weight_decay_sum
        self.config['t2i_sim_loss'] = tf.reduce_mean(
            tf.maximum(0.0, self.config['delta'] + self.config['rel_distance'] - self.config[
                't2i_irrel_distance'])) + weight_decay_sum

        global_step = tf.Variable(0, trainable=False)
        lr_step = tf.train.exponential_decay(self.config['learning_rate'], global_step, 20000, 0.95, staircase=True)
        self.config['i2t_optimizer'] = tf.train.GradientDescentOptimizer(lr_step)
        self.config['i2t_updates'] = self.config['i2t_optimizer'].minimize(self.config['i2t_sim_loss'])

        self.config['t2i_optimizer'] = tf.train.GradientDescentOptimizer(lr_step)
        self.config['t2i_updates'] = self.config['t2i_optimizer'].minimize(self.config['t2i_sim_loss'])
