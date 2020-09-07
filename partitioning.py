import pandas as pd
import numpy as np
import math
import collections
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.cluster.vq import kmeans2
import random 
from tqdm import tqdm_notebook
from scipy.spatial.distance import cdist, squareform
from sklearn.metrics import silhouette_samples
from random import randint
from evaluation import l2_distance
import operator
import time
import tqdm

def hierarchical_partition(data, cluster_size, method='ward'):
    data_frame = pd.DataFrame(data)
    #all_distances = pd.DataFrame(cdist(data_frame.values, data_frame.values))
    number_clusters = math.floor(len(data) / cluster_size)
    cluster_size = math.floor(len(data) / number_clusters )
    #print('number_clusters ', number_clusters)
    cluster_lengths = [cluster_size] * number_clusters
    #make sure that the last points are clustered
    for clt in range(len(data) % cluster_size):
        cluster_lengths[clt] += 1
    #print(cluster_lengths)

    labels = [-1] * len(data)
    counter = 0
    cut_slices = 0
    for cluster_length in tqdm_notebook(cluster_lengths[1:], desc='[Clustering]: ',disable=False):
    #for cluster_length in cluster_lengths[1:]:
        #print('cluster_length ', cluster_length)
        unclustered_indices = [i for i, x in enumerate(labels) if x == -1]
        unclustered_data = data_frame.iloc[unclustered_indices, ]
        Z = linkage(unclustered_data, method = method)
        current_cluster_counts = collections.Counter([0])
        number_clusters = math.ceil(len(unclustered_indices) / cluster_size)
        cut_slices = number_clusters * 2
    
        labels_cutree = []

        while max(current_cluster_counts.values()) < cluster_length:
            labels_cutree = cut_tree(Z, cut_slices)
            current_cluster_counts = collections.Counter(labels_cutree.T[0])
            cut_slices -= 1

        #print('counter ', counter)
        if counter == 0:
            optimal_cut = cut_slices
        #(name, size) of the biggest cluster
        biggest_cluster = current_cluster_counts.most_common(1)
        #cluster number corresponding to the above size is: biggest_cluster[0][0]
        found_indices = [i for i, x in enumerate(labels_cutree) if x == biggest_cluster[0][0]]
        #extract the first cluster_length points
        found_indices = found_indices[:cluster_length]
        # update the labels vector
        for index in found_indices:
            labels[unclustered_indices[index]] = counter
        counter += 1

    outliers = [i for i, x in enumerate(labels) if x == -1]

    #print('outliers ', outliers)
    for outlier in outliers:
        labels[outlier] = counter


    return labels  

def kmeans_partition(data, cluster_size = 2, method ='elki'):
    df = pd.DataFrame(data)
    number_clusters = math.floor(len(data) / cluster_size)
    # print('number_clusters: ', number_clusters)

    cluster_size = math.floor(len(data) / number_clusters)

    centroid, label = kmeans2(data, number_clusters, minit='points')
    df_centroid = pd.DataFrame(centroid)
    labels = [-1] * len(data)
    cluster_lengths = [0] * number_clusters
    target_size = [cluster_size] * number_clusters
    
    for clt in range(len(data) % cluster_size):
        target_size[clt] += 1
    # print('target_size ', target_size)
    df_dist = df_centroid.apply(lambda row: \
                            (df - row).apply(np.linalg.norm, axis = 1), axis = 1)
    df_dist = df_dist.transpose()
    if method == 'elki':
        df_dist_delta = df_dist.apply(lambda row: row.min()-row.max(), axis =1)
        order_idx = np.argsort(df_dist_delta, axis=1)
    elif method == 'max':
        df_dist_max = df_dist.max(axis=1)
        order_idx = np.argsort(-df_dist_max, axis=1)
    elif method == 'min':
        df_dist_min = df_dist.min(axis=1)
        order_idx = np.argsort(df_dist_min, axis=1)
    elif method == 'random':
        order_idx = random.sample(range(len(data)), len(data))
    for i, idx in enumerate(order_idx):
        best_cluster = np.argmax(df_dist.iloc[idx,])
        labels[idx] = best_cluster
        cluster_lengths[best_cluster] += 1
        if cluster_lengths[best_cluster] >= target_size[best_cluster]:
            df_dist[best_cluster] = np.NaN
    
    return labels

def kNN_partition(data, cluster_size = 2, method = 'max'):
    df = pd.DataFrame(data)
    number_clusters = math.floor(len(data) / cluster_size)
    #print('number_clusters ', number_clusters)
    cluster_size = math.floor(len(data) / number_clusters)
    target_size = [cluster_size] * number_clusters
    #print('target_size ',target_size)
    for clt in range(len(data) % cluster_size):
        target_size[clt] += 1
    labels = [-1] * len(data)
    dist_matrix = pd.DataFrame(cdist(data, data))
    counter = 0

    while -1 in labels:
        unclustered_indices = [i for i, x in enumerate(labels) if x == -1]
        unclustered_matrix = dist_matrix.loc[unclustered_indices, unclustered_indices]
        if method == 'max':
            idx = np.argmax(unclustered_matrix.sum(axis = 1))
        elif method == 'min':
            idx = np.argmin(unclustered_matrix.sum(axis = 1))
        elif method == 'random':
            idx = randint(0, len(unclustered_indices) - 1 )

        temp_cluster = [-1] * len(unclustered_indices)

        if len(unclustered_matrix.iloc[idx]) >= target_size[counter]:
            closest_idx = unclustered_matrix.iloc[idx].argsort()[0:target_size[counter]]
            # closest_idx =  np.argpartition(unclustered_matrix.iloc[idx], -target_size[counter])[-target_size[counter]:]
        else:
            closest_idx = unclustered_matrix.iloc[idx]
        for item in closest_idx:
            temp_cluster[item] = counter
        for i, item in enumerate(unclustered_indices):
            labels[item] = temp_cluster[i]
        counter += 1
        
    return labels

def fixed_kmeans(min_size, n_sample, data):
  if n_sample % min_size == 0:
      k_cluster = n_sample /i
      max_size = min_size
  else:
      k_cluster = math.floor(n_sample/min_size)
      max_size = min_size + 1
  #clustering ...
  clusters, centers = minsize_kmeans(data, k_cluster, min_size, max_size)
  return clusters

def random_partition(data, k):
    number_clusters = len(data) // k
    cluster_size = len(data) // number_clusters
    target_size = [cluster_size] * number_clusters
    labels = [-1] * len(data)
    for clt in range(len(data) % cluster_size):
        target_size[clt] += 1
    indices = [i for i,_ in enumerate(data)]
    for clt in range(number_clusters):
        cluster_mem = random.sample(indices, target_size[clt])
        indices = list( set(indices) - set(cluster_mem) )
        for mem in cluster_mem:
            labels[mem] = clt
    return labels

def cluster_center(cluster):
    return np.mean(cluster).tolist()

def m_adjust_cluster(cluster, residual, k):
    center = cluster_center(cluster)
    dist_dict = {}
    # add random seed to cluster
    for i, t in enumerate(cluster):
        dist = l2_distance(center, t)
        dist_dict[i] = dist
    sorted_dict = sorted(dist_dict.items(), key=operator.itemgetter(1))
    need_adjust_index = [t[0] for t in sorted_dict[k:]]
    need_adjust = [cluster[t] for t in need_adjust_index]
    residual.extend(need_adjust)
    # update cluster
    #print('Need adjust len: ', len(need_adjust))
    cluster = [t for i, t in enumerate(cluster)
                      if i not in set(need_adjust_index)]
    #print('cluster len: ', len(cluster))
    return cluster, residual

def m_find_best_cluster_iloss(record, clusters):
    """residual assignment. Find best cluster for record."""
    min_distance = 1000000000000
    min_index = 0
    best_cluster = clusters[0]
    for i, t in enumerate(clusters):
        #print('i of less clt: ', i)
        if not t == -1:
            centroid = cluster_center(t)
            #print('centroid: ', centroid)
            distance = l2_distance(record, centroid)
    #         print('distance: ', distance)
            if distance < min_distance:
                min_distance = distance
    #             print('min_distance: ', min_distance)
                min_index = i
                best_cluster = t
    #             print('min_index: ', i)
        # add record to best cluster
    #     print('output min index: ', min_index)
    return min_index

def m_find_furthest_record(record, data):
    """
    :param record: the latest record be added to cluster
    :param data: remain records in data
    :return: the index of the furthest record from r_index
    """
    max_distance = 0
    max_index = -1
    for index in range(len(data)):
        current_distance = l2_distance(record, data[index])
        if current_distance >= max_distance:
            max_distance = current_distance
            max_index = index
    return max_index

def m_find_best_record_iloss_increase(cluster, data):
    """
    :param cluster: current
    :param data: remain dataset
    :return: index of record with min diff on information loss
    """
    # pdb.set_trace()
    min_diff = 1000000000000
    min_index = 0
    for index, record in enumerate(data):
        # IF_diff = diff_distance(record, cluster)
        # IL(cluster and record) and |cluster| + 1 is a constant
        # so IL(record, cluster.gen_result) is enough
        #t = time.time()
        IF_diff = m_diff_distance(record, cluster)
        #print('Time for a single diff_distance: ', time.time()-t)
        if IF_diff < min_diff:
            min_diff = IF_diff
            min_index = index
    return min_index

def m_find_best_cluster_iloss_increase(record, clusters):
    """residual assignment. Find best cluster for record."""
    min_diff = 1000000000000
    min_index = 0
    best_cluster = clusters[0]
    for i, t in enumerate(clusters):
        
        IF_diff = m_diff_distance(record, t)
        
        if IF_diff < min_diff:
            min_distance = IF_diff
            min_index = i
            best_cluster = t
    # add record to best cluster
    return min_index

def m_diff_distance(record, cluster):
    new_cluster = cluster + [record]
    #t=time.time()
    new_IL = IL(new_cluster)
    #print('Time for IL: ', time.time() - t)
    return new_IL - IL(cluster)

def get_index(cluster, data):
    index = [i for i,d in enumerate(data) if d in cluster ]
    return index

def IL(cluster):
    cluster_distances = cdist(cluster,cluster)
    # idx = get_index(cluster, data)
    # cluster_distances = dist_mat.loc[idx,idx]
    return np.array(cluster_distances).max()

def kmember_partition(data, k=20, verbose=False):
    """
    Group record according to NCP. K-member
    """
    clusters = []
    # randomly choose seed and find k-1 nearest records to form cluster with size k
    r_pos = random.randrange(len(data)) # select a single point
    r_i = data[r_pos]
    while len(data) >= k:
        t = time.time()
        r_pos = m_find_furthest_record(r_i, data)
        r_i = data.pop(r_pos)
        cluster = [r_i]
        while len(cluster) < k:
            #t = time.time()
            r_pos = m_find_best_record_iloss_increase(cluster, data)
            #print('Time for finding best record: ', time.time() - t)
            r_j = data.pop(r_pos)
            #print('r_j ', r_j)
            cluster.append(r_j)
            #print(cluster)
        clusters.append(cluster)
        if verbose==True:
            print('Time for finding this cluster: ', time.time() - t)
            print('Current remaining datapoints: ', len(data))
    # residual assignment
    while len(data) > 0:
        t = data.pop()
        cluster_index = m_find_best_cluster_iloss_increase(t, clusters)
        clusters[cluster_index].append(t)
    return clusters

def oka_partition(data, k=20):
    """
    Group record according to NCP. OKA: one time pass k-means
    """
    clusters = []
    can_clusters = []
    less_clusters = []
    n_cluster = int(math.floor(len(data)/k))
    # randomly choose seed and find k-1 nearest records to form cluster with size k
    seed_index = random.sample(range(len(data)), n_cluster)
    for index in seed_index:
        record = data[index]
        can_clusters.append([record])

    data = [t for i, t in enumerate(data) if i not in set(seed_index)]
    # pdb.set_trace()
    while len(data) > 0:
        record = data.pop()
        #print('record: ', record)
        index = m_find_best_cluster_iloss(record, can_clusters)
        #print('index: ', index)
        can_clusters[index].append(record)
    residual = []
    residuals = []
    less_clusters = [-1] * len(can_clusters) # make sure that less_clusters has all clusters
    for i, cluster in enumerate(can_clusters):
        if len(cluster) < k:
            less_clusters[i] = cluster
        else:
            if len(cluster) > k:
                cluster, residual = m_adjust_cluster(cluster, residual, k)
            clusters.append(cluster)
    n_less_clusters = 0
    for clt in less_clusters:
        if clt != -1:
            n_less_clusters += 1
    while len(residual) > 0:
        record = residual.pop()
        if n_less_clusters > 0:
            index = m_find_best_cluster_iloss(record, less_clusters)
            #print('found index: ', index)
            less_clusters[index].append(record)
            #print('len of less clt: ', len(less_clusters[index]))
            if len(less_clusters[index]) >= k: #if the less cluster is full
                clusters.append(less_clusters[index]) #add it to the result
                n_less_clusters -= 1
                less_clusters[index] = -1
        else:
            index = m_find_best_cluster_iloss(record, clusters)
            clusters[index].append(record)
    return clusters

def cluster_member_to_index(clusters_member, data):
    labels = [-1] * len(data)
    for i, cluster in enumerate(clusters_member):
        #print('len: ', len(cluster))
        idx = [j for j,d in enumerate(data) if d in cluster]
        #print('index: ', idx)
        for item in idx:
            labels[item] = i
    return labels

def cluster_stats(data, labels):
    dist_matrix = pd.DataFrame(cdist(data,data))
    silhouette_values = silhouette_samples(data, labels)
    labelList = np.unique(labels)
    df = pd.DataFrame()
    for label in labelList:
        cluster_indices = [i for i,x in enumerate(labels) if x==label]
        if len(cluster_indices) == 1:
            cluster_dist_matrix = 0
        else:
            cluster_dist_matrix = squareform(dist_matrix.iloc[cluster_indices, cluster_indices])
        min_dist  = np.min(cluster_dist_matrix)
        max_dist  = np.max(cluster_dist_matrix)
        mean_dist = np.mean(cluster_dist_matrix)
        mean_silhouette = silhouette_values[np.array(labels) == label].mean()
        cluster_record = pd.DataFrame(dict(
            label=(label),
            size=(len(cluster_indices)),
            min_dist=(min_dist),
            max_dist=(max_dist),
            mean_dist=(mean_dist),
            mean_silhouette=(mean_silhouette) ), index=[0])
        df = df.append(cluster_record, ignore_index=True)
    return df

def hp_v2(data,k,method='ward'):
    tree = linkage(data,method)
    clusters = partitioning_tree(k,len(data),tree)
    return clusters

def partitioning_tree(k,N,tree):
    clusters = [-1] * N
    anchor  = get_anchor(k, tree[:,3])
    counter = 0
    while len(anchor) !=0:
        #print('This is anchor: ', anchor)
        for i,size in enumerate(tree[:,3]):
            if size >= k:
                clusters = update_cluster(i, N, clusters, counter, tree.copy())
                #print('This is cluster: ', clusters)
                tree = update_tree(i,N, tree.copy())
                #print('This is tree:\n ', tree)
                counter += 1
                anchor  = get_anchor(k, tree[:,3])
                break
    return clusters

def get_anchor(k, sizes):
    anchor = [x for x in sizes if x >= k]
    return anchor

def update_cluster(idx, N, clusters, counter,tree):
    mems = get_mems(idx, N, tree)
    for i in mems:
        clusters[i] = counter
    return clusters

def get_mems(idx, N, tree):
    mems = []
    for i in [ tree[idx,0], tree[idx,1] ]:
        if i >= N and tree[int(i-N),3] != 0:
            mems.extend( get_mems( int(i-N),N, tree) )
        elif i < N:
            mems.append(int(i))
    return mems

def update_tree(i,N,tree):
    tree = update_fw(i,N,tree)
    tree = update_bw(i,N,tree)
    return tree

def update_bw(idx, N, tree):
    for i in [ tree[idx,0], tree[idx,1] ]:
        tree[idx,3] = 0
        if i >= N:
            tree = update_bw(int(i-N), N, tree)
    return tree

def update_fw(idx,N,tree):
    C = idx + N
    for i in range(idx+1, len(tree)):
        prv_idx = idx
        if C in [ tree[i,0], tree[i,1] ]:
            tree[i,3] -= tree[prv_idx,3]
            prv_idx = i
            C = i + N
    return tree