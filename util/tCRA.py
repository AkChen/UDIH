import numpy as np
from CRA import CRA
import scipy.io as scio
import tensorflow as tf
import random
import os
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

test_partial_full = np.asarray([[0.1, 0.2, 0.3, 0, 0, 0], [0, 0, 0, 0.7, 0.8, 0.9]])
test_full = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])

sess = tf.Session()
gvi = tf.initialize_all_variables()
cra = CRA(sess, 6, 4, [6, 4, 4, 6], ['dummy', 'tanh', 'tanh', 'tanh'], 0.02)
# cra_recover_config = np.load('../model/cra.npy')
# cra =  CRA(sess,recover_config=cra_recover_config)
sess.run(gvi)
cra.train(sess, test_partial_full, test_full, 500, 2, 20, show_CRA_epoch=1, show_RA_epoch=0, show_CRA_loss=True)
pre = cra.predict_full(sess, test_partial_full)
print(pre)
# cra.save_model(sess,'../model/cra')

print('loda')
