import tensorflow as tf
import numpy as np
from PCMH_Hashing import PCMH
import os.path
import random
from map import MAP

WORK_DIR = '../data/MIR/'

FEATURE_DIR = WORK_DIR + 'feature/'
LIST_DIR = WORK_DIR + 'list/'

K = 1
IMAGE_DIM = 150
TEXT_DIM = 500
IMAGE_HIDDEN_DIM = 8192
TEXT_HIDDEN_DIM = 8192
BIT = 16
ALPHA = 1.0
BETA = 0.4
GAMMA = 0.4
DELTA = 2.0
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
SAMPLE_EPOCH = 40
SHOWMAP_EPCH = 2 # 2

EPOCH = 1000 #1000
CUR_DATASET = 'MIR'
BATCH_SIZE = 64

MAP_RECORD_FILE = '../record/map_' + CUR_DATASET + '_' + str(BIT) + '.txt'
CONV_RECORD_FILE = '../record/conv_' + CUR_DATASET + '_' + str(BIT) + '.txt'

loss_with_epoch = []
map_with_epoch = []

exist_batch_one_mul = np.ones(BATCH_SIZE)
exist_batch_zero_mul = np.zeros(BATCH_SIZE)


def sigmoid(val):
    s = 1. / (1. + np.exp(val))
    return s


map_min_max = 0.


def load_train_data():
    # step = train,test,val
    feature_data = np.load(FEATURE_DIR + 'train_cra_data_.npy')
    image_list = np.load(LIST_DIR + 'train_image_list_.npy')
    image_feature = feature_data[:, 0:IMAGE_DIM]
    text_feature = feature_data[:, IMAGE_DIM:IMAGE_DIM + TEXT_DIM]
    text_list = np.load(LIST_DIR + 'train_text_list_.npy')
    labels = np.load(LIST_DIR + 'train_label_.npy')

    cross_knn = []  # firstK = image,secondK = text
    for i in range(image_feature.shape[0]):
        cross_knn.append([i, i])

    return image_list, image_feature, text_list, text_feature, labels, cross_knn


def load_step_data(step):
    feature_data = np.load(FEATURE_DIR + step + '_partial_data.npy')
    image_list = np.load(LIST_DIR + step + '_image_list.npy')
    text_list = np.load(LIST_DIR + step + '_text_list.npy')
    step_label = np.load(LIST_DIR + step + '_label.npy')

    # return
    rt_image_feature = []
    rt_image_list = []
    rt_image_label = []
    rt_text_feature = []
    rt_text_list = []
    rt_text_label = []
    for i in range(len(feature_data)):
        if image_list[i].endswith('_p'):  # add text
            rt_text_feature.append(feature_data[i][IMAGE_DIM:IMAGE_DIM + TEXT_DIM])
            rt_text_list.append(text_list[i])
            rt_text_label.append(step_label[i])
        else:
            if text_list[i].endswith('_p'):
                rt_image_feature.append(feature_data[i][:IMAGE_DIM])
                rt_image_list.append(image_list[i])
                rt_image_label.append(step_label[i])
            else:
                rt_text_feature.append(feature_data[i][IMAGE_DIM:IMAGE_DIM + TEXT_DIM])
                rt_text_list.append(text_list[i])
                rt_text_label.append(step_label[i])
                rt_image_feature.append(feature_data[i][:IMAGE_DIM])
                rt_image_list.append(image_list[i])
                rt_image_label.append(step_label[i])

    return np.asarray(rt_image_list), np.asarray(rt_image_feature), np.asarray(rt_image_label), np.asarray(
        rt_text_list), np.asarray(rt_text_feature), np.asarray(rt_text_label)


def load_exist_data():
    image_data = np.load(FEATURE_DIR + 'train_exist_image.npy')
    text_data = np.load(FEATURE_DIR + 'train_exist_text.npy')
    return image_data,text_data


def get_pair_multiplier(namea, nameb):  # adis = a到object的距离
    rt_mul = 1.0
    if namea.endswith('_p'):
        if nameb.endswith('_p'):
            rt_mul = GAMMA
        else:
            rt_mul = BETA
    else:
        if nameb.endswith('_p'):
            rt_mul = BETA
        else:
            rt_mul = ALPHA
    # 考虑距离

    # rt_mul = rt_mul * np.exp(-1 * adis) * np.exp(-1 * bdis)
    return rt_mul


def get_l2_dis(a, b):
    return np.sqrt(np.sum(np.square(a - b)))




# load train  data
print('loading train data')
train_image_list, train_image_feature, train_text_list, train_text_feature, train_labels, train_cross_knn = load_train_data()
# load test and val data
print('loading test data')
test_image_list, test_image_feature, test_image_label, test_text_list, test_text_feature, test_text_label = load_step_data(
    'test')
print('loading val data')
val_image_list, val_image_feature, val_image_label, val_text_list, val_text_feature, val_text_label = load_step_data(
    'val')



def sample_train_data(image_list, image_feature, text_list, text_feature, cross_knn, flag):  # flag = i2t or t2i
    rel_mul = []  # multiplier
    irrel_mul = []  # multiplier
    org_feature = []
    dst_rel_feature = []
    dst_irrel_feature = []

    if flag == 'i2t':
        for idx in range(len(image_list)):
            cur_knn = cross_knn[idx]
            # rel pair
            org_img_idx = cur_knn[random.sample(range(K), k=1)[0]]
            dst_txt_idx = cur_knn[K + random.sample(range(K), k=1)[0]]
            m = get_pair_multiplier(image_list[org_img_idx], text_list[dst_txt_idx])
            rel_mul.append(m)
            org_feature.append(image_feature[org_img_idx])
            dst_rel_feature.append(text_feature[dst_txt_idx])
            # irrel pair
            filter_txt = cur_knn[K:]
            rs = random.sample(range(len(text_list)), k=1)[0]

            while filter_txt.__contains__(rs):
                rs = random.sample(range(len(text_list)), k=1)[0]

            m = get_pair_multiplier(image_list[org_img_idx], text_list[rs])
            irrel_mul.append(m)
            dst_irrel_feature.append(text_feature[rs])

            # sample exist data

        return np.asarray(org_feature), np.asarray(dst_rel_feature), np.asarray(dst_irrel_feature), rel_mul, irrel_mul

    else:
        if flag == 't2i':
            for idx in range(len(text_list)):
                cur_knn = cross_knn[idx]
                # rel pair
                org_txt_idx = cur_knn[K + random.sample(range(K), k=1)[0]]
                dst_img_idx = cur_knn[random.sample(range(K), k=1)[0]]

                m = get_pair_multiplier(text_list[org_txt_idx], image_list[dst_img_idx])
                rel_mul.append(m)
                org_feature.append(text_feature[org_txt_idx])
                dst_rel_feature.append(image_feature[dst_img_idx])
                # irrel pair
                filter_img = cur_knn[:K]

                rs = random.sample(range(len(image_list)), k=1)[0]
                while filter_img.__contains__(rs):
                    rs = random.sample(range(len(image_list)), k=1)[0]
                    # rs = np.random.choice(range(len(image_list)), 1, True, p=prob)[0]
                b_dis = 0.0  # get_l2_dis(image_feature[rs],image_feature[idx])
                m = get_pair_multiplier(text_list[org_txt_idx], image_list[rs])
                irrel_mul.append(m)
                dst_irrel_feature.append(image_feature[rs])

                # sample exist data


            return np.asarray(org_feature), np.asarray(dst_rel_feature), np.asarray(
                dst_irrel_feature), rel_mul, irrel_mul





print('creating PCMH')
pcmh = PCMH(IMAGE_DIM, IMAGE_HIDDEN_DIM, TEXT_DIM, TEXT_HIDDEN_DIM, BIT, LEARNING_RATE, DELTA, WEIGHT_DECAY, None)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('start train')

# 清理文件
if os.path.exists(MAP_RECORD_FILE):
    os.remove(MAP_RECORD_FILE)

for e in range(EPOCH):
    print('EPOCH:%d' % e)
    if e % SAMPLE_EPOCH == 0:
        print('start sampling')
        i2t_org, i2t_rel, i2t_irrel, i2t_rel_mul, i2t_irrel_mul= sample_train_data(train_image_list,
                                                                                    train_image_feature,
                                                                                    train_text_list, train_text_feature,
                                                                                    train_cross_knn, 'i2t',)
        t2i_org, t2i_rel, t2i_irrel, t2i_rel_mul, t2i_irrel_mul = sample_train_data(train_image_list,
                                                                                    train_image_feature,
                                                                                    train_text_list, train_text_feature,
                                                                                    train_cross_knn, 't2i')
    cur_index = 0
    i2t_len = len(i2t_org)
    while cur_index < i2t_len:
        if cur_index + BATCH_SIZE < i2t_len:
            cur_i2t_org = i2t_org[cur_index:cur_index + BATCH_SIZE]
            cur_i2t_rel = i2t_rel[cur_index:cur_index + BATCH_SIZE]
            cur_i2t_irrel = i2t_irrel[cur_index:cur_index + BATCH_SIZE]
            cur_i2t_rel_mul = i2t_rel_mul[cur_index:cur_index + BATCH_SIZE]
            cur_i2t_irrel_mul = i2t_irrel_mul[cur_index:cur_index + BATCH_SIZE]
        else:
            cur_i2t_org = i2t_org[cur_index:i2t_len]
            cur_i2t_rel = i2t_rel[cur_index:i2t_len]
            cur_i2t_irrel = i2t_irrel[cur_index:i2t_len]
            cur_i2t_rel_mul = i2t_rel_mul[cur_index:i2t_len]
            cur_i2t_irrel_mul = i2t_irrel_mul[cur_index:i2t_len]

        _ = sess.run(pcmh.config['i2t_updates'], feed_dict={pcmh.config['image_rel_input']: cur_i2t_org,
                                                            pcmh.config['text_rel_input']: cur_i2t_rel,
                                                            pcmh.config['text_irrel_input']: cur_i2t_irrel,
                                                            pcmh.config['rel_mul']: cur_i2t_rel_mul,
                                                            pcmh.config['irrel_mul']: cur_i2t_irrel_mul})
        i2t_loss = sess.run(pcmh.config['i2t_sim_loss'], feed_dict={pcmh.config['image_rel_input']: cur_i2t_org,
                                                                    pcmh.config['text_rel_input']: cur_i2t_rel,
                                                                    pcmh.config['text_irrel_input']: cur_i2t_irrel,
                                                                    pcmh.config['rel_mul']: cur_i2t_rel_mul,
                                                                    pcmh.config['irrel_mul']: cur_i2t_irrel_mul})
        cur_index += BATCH_SIZE
    print('i2t_loss:%.4f' % i2t_loss)

    cur_index = 0
    t2i_len = len(t2i_org)
    while cur_index < t2i_len:
        if cur_index + BATCH_SIZE < t2i_len:
            cur_t2i_org = t2i_org[cur_index:cur_index + BATCH_SIZE]
            cur_t2i_rel = t2i_rel[cur_index:cur_index + BATCH_SIZE]
            cur_t2i_irrel = t2i_irrel[cur_index:cur_index + BATCH_SIZE]
            cur_t2i_rel_mul = t2i_rel_mul[cur_index:cur_index + BATCH_SIZE]
            cur_t2i_irrel_mul = t2i_irrel_mul[cur_index:cur_index + BATCH_SIZE]
        else:
            cur_t2i_org = t2i_org[cur_index:t2i_len]
            cur_t2i_rel = t2i_rel[cur_index:t2i_len]
            cur_t2i_irrel = t2i_irrel[cur_index:t2i_len]
            cur_t2i_rel_mul = t2i_rel_mul[cur_index:t2i_len]
            cur_t2i_irrel_mul = t2i_irrel_mul[cur_index:t2i_len]

        _ = sess.run(pcmh.config['t2i_updates'], feed_dict={pcmh.config['text_rel_input']: cur_t2i_org,
                                                            pcmh.config['image_rel_input']: cur_t2i_rel,
                                                            pcmh.config['image_irrel_input']: cur_t2i_irrel,
                                                            pcmh.config['rel_mul']: cur_t2i_rel_mul,
                                                            pcmh.config['irrel_mul']: cur_t2i_irrel_mul})
        i2t_loss = sess.run(pcmh.config['t2i_sim_loss'], feed_dict={pcmh.config['text_rel_input']: cur_t2i_org,
                                                                    pcmh.config['image_rel_input']: cur_t2i_rel,
                                                                    pcmh.config['image_irrel_input']: cur_t2i_irrel,
                                                                    pcmh.config['rel_mul']: cur_t2i_rel_mul,
                                                                    pcmh.config['irrel_mul']: cur_t2i_irrel_mul})
        cur_index += BATCH_SIZE
    print('t2i_loss:%.4f' % i2t_loss)

    if (e + 1) % SHOWMAP_EPCH == 0:
        test_i_hash = sess.run(pcmh.config['image_rel_hash'],
                               feed_dict={pcmh.config['image_rel_input']: test_image_feature})
        test_t_hash = sess.run(pcmh.config['text_rel_hash'],
                               feed_dict={pcmh.config['text_rel_input']: test_text_feature})
        val_i_hash = sess.run(pcmh.config['image_rel_hash'],
                              feed_dict={pcmh.config['image_rel_input']: val_image_feature})
        val_t_hash = sess.run(pcmh.config['text_rel_hash'], feed_dict={pcmh.config['text_rel_input']: val_text_feature})
        i2tmap = MAP(test_i_hash, test_image_label, val_t_hash, val_text_label)
        t2imap = MAP(test_t_hash, test_text_label, val_i_hash, val_image_label)
        i2imap = MAP(test_i_hash, test_image_label, val_i_hash, val_image_label)
        t2tmap = MAP(test_t_hash, test_text_label, val_t_hash, val_text_label)
        print('I2TMAP:%.4f' % i2tmap)
        print('T2IMAP:%.4f' % t2imap)
        print('I2IMAP:%.4f' % i2imap)
        print('T2TMAP:%.4f' % t2tmap)
        loss_with_epoch.append((e, i2t_loss))
        map_with_epoch.append((e, i2tmap, t2imap))
        if np.min([i2tmap, t2imap]) > map_min_max:
            map_min_max = np.min([i2tmap, t2imap])
            map_rd = open(MAP_RECORD_FILE, 'a+')
            map_rd.write('EPOCH:%d\n' % e)
            map_rd.write('I2TMAP:%.4f\n' % i2tmap)
            map_rd.write('T2IMAP:%.4f\n' % t2imap)
            map_rd.close()

converge = open(CONV_RECORD_FILE, 'a+')
str1 = ''
for e in range(EPOCH):
    str1 = str1 + ',' + str(e)
converge.write(str1 + '\n')
str1 = ''
for e in loss_with_epoch:
    str1 = str1 + ',' + str(e[1])
converge.write(str1 + '\n')
str1 = 'I2T:'
for e in map_with_epoch:
    str1 = str1 + ',' + str(e[1])
converge.write(str1 + '\n')
str1 = 'T2I:'
for e in map_with_epoch:
    str1 = str1 + ',' + str(e[2])
converge.write(str1 + '\n')
