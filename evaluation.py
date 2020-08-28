from utils import cluster_dict
import face_recognition
import pandas as pd
#import partitioning
import numpy as np

# from clustering.algorithms.accuracy import fakeid_quality, l2_distance

def dist_to_fakeid(img, 
                   cluster_id, 
                   clusters, 
                   data, 
                   df_stats
                   ):
    boxes = face_recognition.face_locations(img, model='cnn')
    fakeid_encoded = face_recognition.face_encodings(img, boxes)
    # print( fakeid_encoded[0] )
    quality_fakeid, distances = fakeid_quality(data, 
                                               clusters, 
                                               fakeid_encoded[0], 
                                               cluster_id)
    df_stats.loc[df_stats.cluster 
                 == cluster_id, 'fakeid_dist'] = \
    df_stats.loc[df_stats.cluster == cluster_id, 
                 'encoding'
                 ].apply(lambda row: l2_distance(row,
                                                 fakeid_encoded[0]))
    
    df_stats.loc[df_stats.cluster == cluster_id, 
                 'information_loss'] = quality_fakeid
    
    df_stats.loc[df_stats.cluster == cluster_id, 
                 'avg_IL'] = \
    (quality_fakeid / len(df_stats[df_stats.cluster 
                                   == cluster_id]))
    
    return quality_fakeid, distances, df_stats

def fakeid_quality(data, cluster_indices, fakeid, cluster_num):
	clusters = cluster_dict(data, cluster_indices)
	if len(clusters) == 0:
		return 0.0
	fakeid_to_member = []
	quality = 0.0
	for encode in clusters[cluster_num]:
		dist = l2_distance(fakeid, encode)
		fakeid_to_member.append(dist)
		quality += dist

	return quality, fakeid_to_member


def l2_distance(point1, point2):
	distance = np.linalg.norm(np.array(point1) - np.array(point2))
	return distance

def cluster_quality(cluster):
	if len(cluster) == 0:
		return 0.0
	quality = 0.0
	for i in range(len(cluster)):
		for j in range(i, len(cluster)):
			quality += l2_distance(cluster[i], cluster[j])
	return quality / len(cluster)

def compute_quality(data, cluster_indices):
	clusters = cluster_dict(data, cluster_indices)
	#print('Cluster keys: ', clusters.keys())
	return sum(cluster_quality(cluster) for cluster in clusters.values())

def compute_centers(clusters, dataset):
# canonical labeling of clusters
	ids = list(set(clusters))
	c_to_id = dict()
	for j, c in enumerate(ids):
		c_to_id[c] = j
	for j, c in enumerate(clusters):
		clusters[j] = c_to_id[c]

	k = len(ids)
	dim = len(dataset[0])
	centers = [[0.0] * dim for i in range(k)]
	counts = [0] * k
	for j, c in enumerate(clusters):
		for i in range(dim):
			centers[c][i] += dataset[j][i]
		counts[c] += 1
	for j in range(k):
		for i in range(dim):
			centers[j][i] = centers[j][i]/float(counts[j])
	return clusters, centers

def l2_normalize(x):
	return x / np.sqrt(np.sum(np.multiply(x, x)))

