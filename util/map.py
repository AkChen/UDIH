from scipy.spatial.distance import cdist
import numpy as np
import tensorflow as tf
import random




def same_cate(label1, label2):
    return (np.sum(np.logical_and(label1, label2)) > 0)


def MAP(src_hash, src_label, dst_hash, dst_label,in_metric ='hamming'):
    dist = cdist(src_hash, dst_hash, metric=in_metric)
    # 计算每个
    map = 0.0
    for idx in range(len(src_hash)):
        sorted_index = np.argsort(dist[idx])[0:50]
        # 计算map
        count = 0
        summap = 0.0
        for didx, dst_idx in enumerate(sorted_index):

            if same_cate(src_label[idx], dst_label[dst_idx]):
                summap += (count + 1) / (didx + 1)
                count += 1
        if count > 0:
            summap = summap / count



       # print(summap)
        #print('e')
        map += summap

    return map / len(src_hash)


""""
src_hash = [[1,0,0,1],[0,0,0,0]]
dst_hash = [[1,0,1,1],[0,0,0,0],[1,0,1,1]]
src_label = [[1,0,0,1],[0,0,0,1]]
dst_label = [[1,0,1,0],[1,1,0,0],[1,1,0,0]]

m = MAP(src_hash,src_label,dst_hash,dst_label)
print(m)
"""""
