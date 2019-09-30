from ResidualAE import RAE
import tensorflow as tf
import numpy as np
import uuid
import sys


class CRA:
    def __init__(self, sess, depth=None, num_of_layers=None, num_per_layer=None, active_function_per_layer=None,
                 learning_rate=None, beta=0.01, loss_type='l2', recover_config=None, ):
        self.config = {}
        self.config['RAEs'] = []
        self.config['depth'] = depth
        if recover_config is None:
            for d in range(0, depth):
                uid = str(uuid.uuid1()) + '_depth_' + str(d)

                rae = RAE(sess, num_of_layers, num_per_layer, active_function_per_layer, learning_rate, RAE_id=uid,
                          input=None, beta=beta, loss_type=loss_type, recover_config=None)

                self.config['RAEs'].append(rae)
        else:
            self.config['depth'] = len(recover_config)
            for rae_config in recover_config:
                rae = RAE(sess, recover_config=rae_config)
                self.config['RAEs'].append(rae)

    def train(self, sess, train_partial_data, train_full_data, test_partial_data, test_full_data, CRA_epoch, RA_epoch,
              batch_size, show_CRA_epoch=0, show_RA_epoch=0,
              show_CRA_loss=0):

        for e in range(CRA_epoch):
            if show_CRA_epoch > 0 and e % show_CRA_epoch == 0:
                if show_CRA_loss:
                    all = self.predict_full(sess, train_partial_data)
                    loss = np.sum(np.square(all - train_full_data))
                    print('CRA epoch:%d train_loss:%f' % (e, loss))
                    if not test_full_data is None:
                        all = self.predict_full(sess, test_partial_data)
                        loss = np.sum(np.square(all - test_full_data))
                        print('CRA epoch:%d test_loss:%f' % (e, loss))
                print('CRA epoch:%d finished' % e)
                sys.stdout.flush()
            # current_index = 0
            # while (current_index < len(train_partial_data)):
            # if current_index + batch_size < len(train_partial_data):
            # current_full_data = train_full_data[current_index:current_index + batch_size]
            # current_partial_data = train_partial_data[current_index:current_index + batch_size]
            # else:
            # current_full_data = train_full_data[current_index:]
            # current_partial_data = train_partial_data[current_index:]
            # first_res = current_full_data - current_partial_data

            # dot not batch,batch in per RA
            # train the 1'st RA
            first_res = train_full_data - train_partial_data
            current_partial_data = train_partial_data
            #train first RA

            self.config['RAEs'][0].train(sess, first_res, epoch=RA_epoch, batch_size=batch_size,
                                         input=current_partial_data,
                                         show_loss_epoch=show_RA_epoch,test_last_full = test_partial_data,test_full = test_full_data)
            last_full_output = self.config['RAEs'][0].predict_full(sess, input=current_partial_data)  # predict full
            last_test_full_output = self.config['RAEs'][0].predict_full(sess,input=test_partial_data)
            for d in range(1, self.config['depth']):
                cur_d_residual = train_full_data - last_full_output
                self.config['RAEs'][d].train(sess,d_residual =  cur_d_residual, epoch=RA_epoch, batch_size=batch_size,
                                             input=last_full_output,
                                             show_loss_epoch=show_RA_epoch,test_last_full = last_test_full_output,test_full = test_full_data)
                last_full_output = self.config['RAEs'][d].predict_full(sess, input=last_full_output)
                last_test_full_output = self.config['RAEs'][d].predict_full(sess, input=last_test_full_output)

    def predict_full(self, sess, partial_data):
        last_full_output = self.config['RAEs'][0].predict_full(sess, input=partial_data)
        for d in range(1, self.config['depth']):
            last_full_output = self.config['RAEs'][d].predict_full(sess, input=last_full_output)
        return last_full_output

    def get_recover_config(self, sess):
        config = []
        for r in self.config['RAEs']:
            config.append(r.get_recover_config(sess))
        return config

    def save_model(self, sess, path):
        save_config = self.get_recover_config(sess)
        np.save(path, save_config)
