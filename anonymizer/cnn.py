#built-in modules
from keras.models import load_model
import os
import numpy as np
import pandas as pd
#self-made modules
from CNN.faces.generate import generate_object
from evaluation import dist_to_fakeid
import partitioning
import sys
import pickle

def load_gnn(model_path='CNN/output/FaceGen.RaFD.model.d5.adam.h5'):
    
    model = load_model(model_path)
    num_ids = model.input_shape[0][1] 
    dataset = os.path.basename(model_path).split('.')[1]
    parameters = ['', 'single', 'neutral']
    parameters.append(model)
    parameters.append(num_ids)
    parameters.append(dataset)
    return parameters

def gnn_fakeids(clusters, adapt_data, adapt_raw_data, 
                parameters=[], output='CNN/results'):
    if len(parameters) == 0:
        parameters = load_gnn() #default parameters
    information_loss = []
    fakeid_dist_avg_gnn =[]
    all_distances_gnn = []
    gnn_ids = gnn_id(clusters, adapt_raw_data)
    labelList = np.unique(clusters)
    data_frame = []
    for i, record in enumerate(adapt_data):
        d = [{'cluster': clusters[i], 'encoding': record}]
        data_frame.extend(d)
    df_stats = pd.DataFrame(data_frame)
    #print(df_stats)
    for cluster_id in gnn_ids.keys():
        #Generating fake ID
        parameters[0] = gnn_ids[cluster_id]
        fakeid = generate_object(parameters, output, batch_size=32, yaml=False)
        
        # calculate distance:
        quality_fakeid, distances, df_stats = dist_to_fakeid(fakeid, cluster_id, clusters, adapt_data, df_stats)
        if type(parameters[0]) is str:
            cluster_ids = parameters[0].split('+')
        else:
            cluster_ids = [parameters[0]]
        #collect all dists from fake ids to clusters
        information_loss.append(quality_fakeid)
        fakeid_dist_avg_gnn.append( quality_fakeid/len(cluster_ids) )
        all_distances_gnn.append(distances)
    df_stats['disclosure'] = df_stats['fakeid_dist'].map(lambda row: 1 if row > 0.6 else 0)
    disclosure_prob = len(df_stats[df_stats['disclosure'] == 0]) / len(clusters)
    return information_loss, fakeid_dist_avg_gnn, all_distances_gnn, gnn_ids.keys(), disclosure_prob, df_stats

def gnn_id(cluster_indices, encoding_data, kid=False):
    id_map = {35: 0, 32: 1, 26: 2, 50: 3, 23: 4, 52: 5, 57: 6, 48: 7, 27: 8, 11: 9, 70: 10, 31: 11, 51: 12, 20: 13, 45: 14, 66: 15, 36: 16, 67: 17, 30: 18, 54: 19, 24: 20, 29: 21, 44: 22, 4: 23, 58: 24, 72: 25, 19: 26, 46: 27, 1: 28, 8: 29, 13: 30, 15: 31, 0: 32, 71: 33, 69: 34, 7: 35, 25: 36, 37: 37, 21: 38, 60: 39, 53: 40, 6: 41, 2: 42, 9: 43, 34: 44, 49: 45, 22: 46, 47: 47, 14: 48, 55: 49, 17: 50, 59: 51, 18: 52, 28: 53, 3: 54, 68: 55, 56: 56}

    cluster_index_dict = dict()
    for i, cluster_index in enumerate(cluster_indices):
        if cluster_index in cluster_index_dict:
            cluster_index_dict[cluster_index].append(i)
        else:
            cluster_index_dict[cluster_index] = [i]
    result = dict()
    count = 0
    for key in sorted(cluster_index_dict):
        ids_string = ''
        
        for index in cluster_index_dict[key]:
            image_path = encoding_data[index]['imagePath']
            if kid == False and 'Kid' not in image_path:
                identity = int(image_path.split('_')[1])-1 
                mapped_id = id_map[identity]
                ids_string += str(mapped_id) + '+'
            elif kid == True:
                identity = int(image_path.split('_')[1])-1 
                mapped_id = id_map[identity]
                ids_string += str(mapped_id) + '+'
        ids_string = ids_string[:-1]
        if '+' not in ids_string and len(ids_string) != 0:
            result[key] = int(ids_string)
        elif not len(ids_string) == 0:
            result[key] = ids_string
    return result

def evaluate_cnn(data, raw_data, clustering=partitioning.hierarchical_partition, k_range=range(2,20)):
    ILs = []
    fail_rates = []

    for size in k_range:
        print('Processing k = ', size)
        #clustering ...
        clusters = clustering(data, size)
        
        information_loss, fakeid_dist_avg_gnn, all_distances_gnn, \
            labelList, disclosure_prob, df_stats \
                = gnn_fakeids(clusters, data, raw_data)
        
        fail_rates.append(disclosure_prob)
        quality = sum(information_loss)
        ILs.append(quality)

    ILs = np.array(ILs)/ len(data)
    return ILs, fail_rates