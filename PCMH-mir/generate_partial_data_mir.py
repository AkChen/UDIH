import numpy as np
import scipy.io as scio
import random
import os
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# org 18000 about 1800 full
TRAIN_DATA_SIZE = 14000  #
TRAIN_PARTIAL_RATE =  0.9# 2000 full
TRAIN_PARTIAL_SIZE = int(TRAIN_DATA_SIZE * TRAIN_PARTIAL_RATE)

TEST_DATA_SIZE = 2000  ## unique to train_data
TEST_PARTIAL_RATE = 0.9
TEST_PARTIAL_SIZE = int(TEST_DATA_SIZE * TEST_PARTIAL_RATE)

VAL_DATA_SIZE = 14000
VAL_PARTIAL_RATE = 0.9
VAL_PARTIAL_SIZE = int(VAL_DATA_SIZE * VAL_PARTIAL_RATE)

OUT_DIR = '../data/mir/'
FEATURE_DIR = OUT_DIR + 'feature/'
LIST_DIR = OUT_DIR + 'list/'


mat_data = scio.loadmat('../data/mirflickr25k.mat')
print(mat_data.keys())
image_data = list(mat_data['I_tr'])
text_data = list(mat_data[
                     'T_tr'])  # np.load('E:\\DataClear\\mirflicker\\tag_vec.npy') #mat_data['text']# mat_data['text'] #np.load('E:\\DataClear\\mirflicker\\tag_vec.npy') #mat_data['text'] #
label_data = list(mat_data['L_tr'])

image_data.extend(list(mat_data['I_te']))
text_data.extend(list(mat_data['T_te']))
label_data.extend(list(mat_data['L_te']))


image_data = np.asarray(image_data)
text_data = np.asarray(text_data)
label_data = np.asarray(label_data)


def process_col(data, col):
    col_data = data[:, col]
    log_data = col_data  # np.log10(col_data)
    log_min = np.min(log_data)
    log_max = np.max(log_data)
    min_max_scale = (log_data - log_min) / (log_max - log_min)
    data[:, col] = min_max_scale


def sigmoid(d):
    return 1.0 / (1. + np.exp(-1.0 * d))


def partial_generate(image, text, label, used_index, random_size, partial_size):
    candidate_index = []
    if not used_index is None:
        for i in range(len(image)):
            if not used_index.__contains__(i):
                candidate_index.append(i)
    else:
        for i in range(len(image)):
            candidate_index.append(i)
    candidate_index_random_index = random.sample(range(0, len(candidate_index)), random_size)
    chossed_data_index = [candidate_index[i] for i in candidate_index_random_index]
    partial_index_index = random.sample(range(0, len(chossed_data_index)), partial_size)

    rt_partial_all_modality = []
    rt_full_all_modality = []
    rt_image_list = []
    rt_text_list = []
    rt_label = []

    for idx, tri in enumerate(chossed_data_index):
        all_mo = []
        full_all_mo = []
        img = image[tri]
        txt = text[tri]

        full_all_mo.extend(img)
        full_all_mo.extend(txt)
        rt_full_all_modality.append(full_all_mo)

        img_name = str(tri) + '.jpg'
        txt_name = str(tri) + '.txt'
        #
        #
        if partial_index_index.__contains__(idx):  # need to partial
            x = random.sample(range(0, 2), 1)[0]
            if x == 0:  # partial image
                img[:] = 0.
                img_name += '_p'
            if x == 1:
                txt[:] = 0.
                txt_name += '_p'
        all_mo.extend(img)
        all_mo.extend(txt)
        rt_partial_all_modality.append(all_mo)
        rt_image_list.append(img_name)
        rt_text_list.append(txt_name)
        rt_label.append(label[tri])
    return chossed_data_index, rt_full_all_modality, rt_partial_all_modality, rt_image_list, rt_text_list, rt_label


used_index = []

test_chossed_index, test_full_all_modality, test_partial_all_modality, test_image_list, test_text_list, test_label = partial_generate(
    image_data, text_data, label_data, used_index, TEST_DATA_SIZE, TEST_PARTIAL_SIZE)
used_index.extend(test_chossed_index)

train_chossed_index, train_full_all_modality, train_partial_all_modality, train_image_list, train_text_list, train_label = partial_generate(
    image_data, text_data, label_data, used_index, TRAIN_DATA_SIZE, TRAIN_PARTIAL_SIZE)
# used_index.extend(train_chossed_index)

val_chossed_index, val_full_all_modality, val_partial_all_modality, val_image_list, val_text_list, val_label = partial_generate(
    image_data, text_data, label_data, used_index, VAL_DATA_SIZE, VAL_PARTIAL_SIZE)
used_index.extend(val_chossed_index)

# save training data
np.save(FEATURE_DIR + 'train_partial_data', train_partial_all_modality)
np.save(FEATURE_DIR + 'train_full_data', train_full_all_modality)
np.save(LIST_DIR + 'train_image_list', train_image_list)
np.save(LIST_DIR + 'train_text_list', train_text_list)
np.save(LIST_DIR + 'train_label', train_label)

np.save(FEATURE_DIR + 'test_partial_data', test_partial_all_modality)
np.save(FEATURE_DIR + 'test_full_data', test_full_all_modality)
np.save(LIST_DIR + 'test_image_list', test_image_list)
np.save(LIST_DIR + 'test_text_list', test_text_list)
np.save(LIST_DIR + 'test_label', test_label)

np.save(FEATURE_DIR + 'val_partial_data', val_partial_all_modality)
np.save(LIST_DIR + 'val_image_list', val_image_list)
np.save(LIST_DIR + 'val_text_list', val_text_list)
np.save(LIST_DIR + 'val_label', val_label)


#for CCQ
ccq_dict = dict()
I_tr = np.asarray(train_full_all_modality)[:,:150]
T_tr = np.asarray(train_full_all_modality)[:,150:]
I_te = np.asarray(test_full_all_modality)[:,:150]
T_te = np.asarray(test_full_all_modality)[:,150:]
L_tr = np.asarray(train_label).astype('int')
L_te = np.asarray(test_label).astype('int')

ccq_dict['I_tr'] = I_tr
ccq_dict['I_te'] = I_te
ccq_dict['T_tr'] = T_tr
ccq_dict['T_te'] = T_te
ccq_dict['L_tr'] = L_tr
ccq_dict['L_te'] = L_te

scio.savemat('H:\\paper\\code\\CCQ\\data\\mir_org.mat',ccq_dict)