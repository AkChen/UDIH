
import sys
sys.path.append("../util")
from CRA import CRA
import numpy as np
import tensorflow as tf
import os
from scipy.spatial.distance import cdist

IMAGE_DIM = 150
TEXT_DIM = 500

P_I = 0.02
P_T = 0.01

CURRENT_DATASET = 'MIR'
CURRENT_MODALITY = CURRENT_DATASET
# load traing data
#
WORK_DIR = '../data/mir/'
FEATURE_DIR = WORK_DIR + 'feature/'
LIST_DIR = WORK_DIR + 'list/'

train_partial_all_modality = np.load(FEATURE_DIR + 'train_partial_data.npy')
train_full_all_modality = np.load(FEATURE_DIR + 'train_full_data.npy')
train_image_list = np.load(LIST_DIR + 'train_image_list.npy')
train_text_list = np.load(LIST_DIR + 'train_text_list.npy')

test_partial_all_modality = np.load(FEATURE_DIR + 'test_partial_data.npy')
test_full_all_modality = np.load(FEATURE_DIR + 'test_full_data.npy')
test_image_list = np.load(LIST_DIR + 'test_image_list.npy')
test_text_list = np.load(LIST_DIR + 'test_text_list.npy')

imin = 0
imax = 0
tmin = 0
tmax = 0


candidate_image_data = []
candidate_text_data = []


def generate_CRA_training_data(full_all, partial_all, image_list, text_list):
    CRA_train_image_full = []
    CRA_train_text_full = []
    CRA_train_image_partial = []  # no image
    CRA_train_text_partial = []  # no text

    # exist both in image and text
    for idx in range(len(image_list)):
        full = full_all[idx]
        if not image_list[idx].endswith('_p') and not text_list[idx].endswith('_p'):
            partial_case_1 = full.copy()
            partial_case_2 = full.copy()
            partial_case_1[0:IMAGE_DIM] = 0.
            partial_case_2[IMAGE_DIM:] = 0.
            CRA_train_image_full.append(full)
            CRA_train_text_full.append(full)
            CRA_train_image_partial.append(partial_case_1)
            CRA_train_text_partial.append(partial_case_2)

            #candidate_image_data.append(full[0:IMAGE_DIM])
            #candidate_text_data.append(full[IMAGE_DIM:])

        if image_list[idx].endswith('_p'):
            candidate_text_data.append(full[IMAGE_DIM:])#text

        if text_list[idx].endswith('_p'):
            candidate_image_data.append(full[0:IMAGE_DIM])



    return np.asarray(CRA_train_image_full), np.asarray(CRA_train_image_partial), np.asarray(
        CRA_train_text_full), np.asarray(CRA_train_text_partial)


CRA_train_image_full, CRA_train_image_partial, CRA_train_text_full, CRA_train_text_partial = generate_CRA_training_data(
    train_full_all_modality, train_partial_all_modality,
    train_image_list, train_text_list)

CRA_test_image_full, CRA_test_image_partial, CRA_test_text_full, CRA_test_text_partial = generate_CRA_training_data(
    test_full_all_modality, test_partial_all_modality,
    test_image_list, test_text_list)

sess = tf.Session()
sess.as_default()

t_v = np.mean(CRA_train_text_full)
if os.path.exists('../model/PCMH_cra_img'+CURRENT_MODALITY+'.npy'):
    cra_recover_config = np.load('../model/PCMH_cra_img'+CURRENT_MODALITY+'.npy')
    cra_image = CRA(sess, recover_config=cra_recover_config)
else:
    # 深度特征大约500次收敛
    cra_image = CRA(sess, 4, 5, [IMAGE_DIM + TEXT_DIM, 1024,512,1024, IMAGE_DIM + TEXT_DIM],
                    # 'dummy', 'tanh', 'sigmoid','sigmoid','sigmoid' 深度特征learning_rate=0.005, beta=0.01,l2,1500
                    # 'dummy', 'sigmoid', 'sigmoid','sigmoid','sigmoid'  0.0005
                    active_function_per_layer=['dummy','sigmoid', 'sigmoid','sigmoid','sigmoid'], learning_rate=0.0001, beta=0.01,
                    loss_type='l2')
    cra_image.train(sess, CRA_train_image_partial, CRA_train_image_full, CRA_test_image_partial, CRA_test_image_full,
                    CRA_epoch=600, RA_epoch=4, batch_size=64,
                    show_CRA_epoch=10,
                    show_RA_epoch=0,
                    show_CRA_loss=True)
    cra_image.save_model(sess, '../model/PCMH_cra_img'+CURRENT_MODALITY)


predict = train_partial_all_modality.copy()
# predict image
for idx in range(len(train_image_list)):
    if train_image_list[idx].endswith('_p'):
        data = np.asarray([train_partial_all_modality[idx]])
        p = cra_image.predict_full(sess, data)[0]
        predict[idx][:IMAGE_DIM] = p[:IMAGE_DIM]


sess.close()
tf.reset_default_graph()

import time

sess = tf.Session()

if os.path.exists('../model/PCMH_cra_txt'+CURRENT_MODALITY+'.npy'):
    cra_recover_config = np.load('../model/PCMH_cra_txt'+CURRENT_MODALITY+'.npy')
    cra_txt = CRA(sess, recover_config=cra_recover_config)
else:
    cra_txt = CRA(sess, 4, 5, [IMAGE_DIM + TEXT_DIM, 1024,512,1024, IMAGE_DIM + TEXT_DIM],
                    active_function_per_layer=['dummy', 'sigmoid', 'sigmoid','sigmoid','sigmoid'], learning_rate=0.0001, beta=0.01,
                  loss_type='l2')

    cra_txt.train(sess, CRA_train_text_partial, CRA_train_text_full, CRA_test_text_partial, CRA_test_text_full,
                  CRA_epoch=600, RA_epoch=4, batch_size=64,
                  show_CRA_epoch=10,
                  show_RA_epoch=0,
                  show_CRA_loss=True)
    cra_txt.save_model(sess, '../model/PCMH_cra_txt'+CURRENT_MODALITY)

for idx in range(len(train_image_list)):
    if train_text_list[idx].endswith('_p'):
        data = np.asarray([train_partial_all_modality[idx]])
        p = cra_txt.predict_full(sess, data)[0]
        predict[idx][IMAGE_DIM:] = p[IMAGE_DIM:]

# sess.close()

#
full_all_index = []
partial_image_index = []
partial_text_index = []

for idx in range(len(train_image_list)):
    if train_text_list[idx].endswith('_p'):
        partial_text_index.append(idx)
    else:
        if train_image_list[idx].endswith('_p'):
            partial_image_index.append(idx)
        else:
            full_all_index.append(idx)

##np.save(FEATURE_DIR + './train_cra_data',predict)


train_label = np.load(LIST_DIR + 'train_label.npy')
new_train_cra_data = []
new_train_image_list = []
new_train_text_list = []
new_train_label = []

import random
print('impute image:%d' % int(len(partial_image_index)*P_I))
print('impute text: %d' % int(len(partial_text_index)*P_T))
_image_index_index = random.sample(range(0, len(partial_image_index)), int(len(partial_image_index)*P_I)) # 补充个数
_text_index_index = random.sample(range(0, len(partial_text_index)),int(len(partial_text_index)*P_T))

# fullfill full
for idx in full_all_index:
    new_train_cra_data.append(predict[idx])
    new_train_label.append(train_label[idx])
    new_train_image_list.append(train_image_list[idx])
    new_train_text_list.append(train_text_list[idx])
for idx in _image_index_index:
    new_train_cra_data.append(predict[partial_image_index[idx]])
    new_train_label.append(train_label[partial_image_index[idx]])
    new_train_image_list.append(train_image_list[partial_image_index[idx]])
    new_train_text_list.append(train_text_list[partial_image_index[idx]])
for idx in _text_index_index:
    new_train_cra_data.append(predict[partial_text_index[idx]])
    new_train_label.append(train_label[partial_text_index[idx]])
    new_train_image_list.append(train_image_list[partial_text_index[idx]])
    new_train_text_list.append(train_text_list[partial_text_index[idx]])

#shuffle
def syc_shuffle(arr_list):
    state = np.random.get_state()
    for e in arr_list:
        np.random.shuffle(e)
        np.random.set_state(state)

syc_shuffle([new_train_cra_data,new_train_image_list,new_train_text_list,new_train_label])

np.save(FEATURE_DIR + './train_cra_data_', new_train_cra_data)
np.save(LIST_DIR + './train_image_list_', new_train_image_list)
np.save(LIST_DIR + './train_text_list_', new_train_text_list)
np.save(LIST_DIR + './train_label_', new_train_label)

