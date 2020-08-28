import pickle
import random
import os
import math
import cv2
import numpy as np
import pandas as pd
import sys

def cluster_idx_dict(clusters):
  cluster_index_dict = dict()
  for i, cluster_index in enumerate(clusters):
      if cluster_index in cluster_index_dict:
          cluster_index_dict[cluster_index].append(i)
      else:
          cluster_index_dict[cluster_index] = [i]
  return cluster_index_dict

def binding_encode_cltId(encodes, clusters):
    result = []
    for i, record in enumerate(encodes):
        d = [{'cluster': clusters[i], 'encoding': record}]
        result.extend(d)
    return pd.DataFrame(result)

def save_result(information_loss, fail_prob, pickle_save_path):
    save_result = pd.DataFrame()
    save_result['IL'] = information_loss
    save_result['FailProb'] = fail_prob
    save_result.to_pickle(pickle_save_path)

def get_mems_of_cluster(cluster_id, clusters):
    mems = [m for m, clt in enumerate(clusters) if clt == cluster_id]
    return mems

def get_filenames(directory, isFolder = False, rafd_kid = True):
    if isFolder:
        if rafd_kid == False:
            filenames = [directory + x for x in os.listdir(directory) if 'Kid' not in x]
        #print(filenames)
        else:  
            filenames = [directory + x for x in os.listdir(directory)]
        filenames.sort()
    else:
        filenames = directory
    return filenames

def order_unique(raw_data_dnn_1000):
    dnn_paths = []
    for d in raw_data_dnn_1000:
        dnn_paths.append(d['imagePath'])
        order_idx = np.argsort(np.array(dnn_paths))
    sorted_raw_dnn_1000 = []
    sorted_dnn_paths = []
    for idx in order_idx:
        sorted_raw_dnn_1000.append(raw_data_dnn_1000[idx])
        sorted_dnn_paths.append(dnn_paths[idx])
    # for d in sorted_raw_dnn_1000:
    #     print(d['imagePath'])
    # print(order_idx)
    import collections
    duplicates_dnn = [item for item, count in collections.Counter(sorted_dnn_paths).items() if count > 1]

    for item in duplicates_dnn:
        print(item)
        elem = [x for x in sorted_raw_dnn_1000 if x['imagePath'] == item]
        sorted_raw_dnn_1000.remove(elem[0])
    print(len(sorted_raw_dnn_1000))
    return sorted_raw_dnn_1000


def get_vectornames(directory, raw_data):
    filenames = []
    ids = [x['imagePath'].split('/')[2].split('.')[0] for x in raw_data ]
    for item in ids:
        filename = directory + item + '_01.npy'
        filenames.append(filename)
    print('Test: ', os.path.isfile(filenames[0]))
    return filenames

def random_list(min_value, max_value, size):
    result = []
    for i in range(size):
        result.append( random.uniform(0, 1) * (max_value - min_value) + min_value )
    return result

def mix_function(vectors, weights):
    product = [a*b for a,b in zip(weights,vectors)]
    mix = sum(product)/len(product)
    return mix

def load_pickle(pickle_path):
    embed_with_meta = pickle.loads( open(pickle_path, 'rb').read() )
    embed_only = [d['encoding'] for d in embed_with_meta]
    return embed_with_meta, embed_only

def save_pickle(data, save_path):
    f = open(save_path, "wb")
    f.write(pickle.dumps(data))
    f.close()

def cluster_dict(data, cluster_indices):
	clusters = dict()
	for i, cluster_index in enumerate(cluster_indices):
		if cluster_index in clusters:
			clusters[cluster_index].append(data[i])
		else:
			clusters[cluster_index] = [data[i]]
	return clusters