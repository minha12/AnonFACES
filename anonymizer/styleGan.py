import os
import numpy as np
import pandas as pd
import sys
import pickle
from tqdm import trange, tqdm_notebook
import matplotlib.pyplot as plt

from utils import get_filenames, cluster_idx_dict, random_list, mix_function
from evaluation import dist_to_fakeid
import partitioning
from vizualization import show_cluster_ids

import styleGAN.dnnlib as dnnlib
import styleGAN.dnnlib.tflib as tflib
sys.modules['dnnlib'] = dnnlib
dnnlib.tflib.init_tf()

def load_styleGan(model_path = 'styleGAN/cache/karras2019stylegan-ffhq-1024x1024.pkl'):
    with dnnlib.util.open_url(model_path) as f:
        generator_network, discriminator_network, averaged_generator_network = pickle.load(f)
    print("StyleGAN loaded & ready for sampling!")
    return [generator_network, discriminator_network, averaged_generator_network] 

def generate_images(generator, latent_vector, z = True):
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, 
                                                  nchw_to_nhwc=True), 
                            minibatch_size=1)
    batch_size = latent_vector.shape[0]
    
    if z: #Start from z: run the full generator network
        return generator.run(latent_vector.reshape((batch_size, 512)), None, 
                                ize_noise=False, **synthesis_kwargs)
    else: #Start from w: skip the mapping network
        return generator.components.synthesis.run(latent_vector.reshape((batch_size, 18, 512)), 
                                                  randomize_noise=False, 
                                                  **synthesis_kwargs)

def cluster_gen(directory, clusters, data, raw_data, generator_network, \
                     isShown = False, rafd_kid = True, alpha=1, 
                     isAdjustWeight = False,
                     isBackward = True,
                     isRandWeight = False,
                     weight_bound = [0.6, 1],
                     k = 100,
                    ):
    '''
    Inputs for this code block:
    - data
    - raw_data for calling 'imagePath'
    - path to latent vectors folder
    - functions:
    + hierarchical_partition
    + generate_images
    + face_recognition.face_locations
    + face_recognition.face_encodings
    + fakeid_quality

    Output for this code block:
    - fakeid_dist_avg
    - all_distances
    '''
    cluster_index_dict = cluster_idx_dict(clusters)
    filenames = get_filenames(directory, isFolder = isBackward)
    data_frame = []
    for i, vectorPath in enumerate(filenames):
        d = [{'cluster': clusters[i], 'encoding': data[i],
          'latentVector': np.load(vectorPath).reshape((1,18,-1))}]
        data_frame.extend(d)
    df_stats = pd.DataFrame(data_frame)
    labelList = np.unique(clusters)
    information_loss = []
    fakeid_dist_avg = []
    all_distances = []

    #for cluster_id in range(448,449):
    for cluster_id in tqdm_notebook(range(0, len(labelList)), desc='[Generating]: '):
    #for cluster_id in labelList:
        ################ print ids in cluster ##########
        if isShown:
            show_cluster_ids(cluster_id, cluster_index_dict, raw_data)
        #################################################
        cluster_vectors = [x['latentVector'] for x in data_frame 
                          if x['cluster'] == cluster_id]
        if isRandWeight:
            weights = random_list(weight_bound[0], weight_bound[1], len(cluster_vectors))
            mix = alpha * mix_function(cluster_vectors, weights)
        else:
            mix = alpha * sum(cluster_vectors)/len(cluster_vectors)
        
        df_stats.loc[df_stats.cluster == cluster_id, 'mix_latent'] = \
            df_stats.loc[df_stats.cluster == cluster_id].apply(lambda row: mix, axis = 1)
        # print(mix.shape)
        img = generate_images(generator_network, mix, z = False)[0]
        quality_fakeid, distances, df_stats = \
             dist_to_fakeid(img, cluster_id, clusters, data, df_stats)
        
        information_loss.append(quality_fakeid)
        fakeid_dist_avg.append( quality_fakeid/len(cluster_vectors) )
        all_distances.append(distances)
        if isShown: 
            plt.imshow(img)
            plt.axis('off')
            plt.title("Generated image for cluster %d" %cluster_id)
            plt.show()
    df_stats['disclosure'] = df_stats['fakeid_dist'].map(lambda row: 1 if row > 0.6 else 0)
    disclosure_prob = len(df_stats[df_stats['disclosure'] == 0]) / len(clusters)
    #Disclosure Risks Assessment
    if isAdjustWeight:
        previous_alpha = alpha
        while disclosure_prob > 1/k:
            current_alpha = previous_alpha*0.8
            df_stats = adjust_weights(df_stats, clusters, data, raw_data, cluster_index_dict, generator_network,
                                    alpha = current_alpha, beta = 0.8, isShown_re_gen = isShown)
            disclosure_prob = len(df_stats[df_stats['disclosure'] == 0]) / len(clusters)
            previous_alpha = current_alpha 
        #calculating below variables according to new df_stats
        information_loss = []
        all_distances = []
        fakeid_dist_avg = []
        for label in labelList:
            cluster_stats = df_stats[df_stats.cluster == label]
            information_loss.append(cluster_stats.information_loss.unique()[0])
            fakeid_dist_avg.append(cluster_stats.avg_IL.unique()[0])
            all_distances.append(cluster_stats.fakeid_dist.values.tolist())
        disclosure_prob = len(df_stats[df_stats['disclosure'] == 0]) / len(clusters)
        
    return fakeid_dist_avg, all_distances, information_loss, \
                labelList, disclosure_prob, df_stats

def adjust_weights(df_stats, clusters, data, raw_data, cluster_index_dict,generator_network, alpha = 0.6, beta = 0.8, isShown_re_gen = False):
    df_disclosure = df_stats[df_stats.disclosure == 0]
    #find which cluster has detected faces
    disclosure_clusters = df_disclosure.cluster.unique()
    for cluster in tqdm_notebook(disclosure_clusters, desc='[Re-generating]: '):
    #for cluster in disclosure_clusters:
        #find detected faces in the cluster
        detected_faces = df_disclosure[df_disclosure.cluster == cluster]
        detected_faces_index = detected_faces.index.values
        #call records belong to the cluster
        df_cluster = df_stats[df_stats.cluster == cluster]
        df_cluster.loc[detected_faces_index, 'latentVector'] = \
            df_cluster.loc[df_cluster.cluster == cluster].apply(lambda row: beta * row)
        #mix the latent vectors
        latent_vectors = df_cluster.latentVector
        mix = alpha * sum(latent_vectors.values) / len(latent_vectors.values)
        #update mix_latent in df_stats
        df_stats.loc[df_stats.cluster == cluster, 'mix_latent'] = \
            df_stats.loc[df_stats.cluster == cluster].apply(lambda row: mix, axis = 1)
        
        img = generate_images(generator_network, mix, z = False)[0]
        if isShown_re_gen:
            print('#################################################')
            print('################ ADJUSTING WEIGHTS ##############')       
            print('#################################################')
            show_cluster_ids(cluster, cluster_index_dict, raw_data)
            plt.imshow(img)
            plt.axis('off')
            plt.title("Re-generated image for cluster %d" %cluster)
            plt.show()
        #re-recalcuate df_stats
        quality_fakeid, distances, df_stats = dist_to_fakeid(img, cluster, clusters, data, df_stats)

    df_stats['disclosure'] = df_stats['fakeid_dist'].map(lambda row: 1 if row > 0.6 else 0)
    return df_stats

def evaluate_styleGan(latents, data, raw_data, model,
                     clustering = partitioning.hierarchical_partition,
                     clt_data = [],
                     k_range = range(2,20),
                     isBackward = False,
                     isShown = False, rafd_kid = True, alpha=1, 
                     isAdjustWeight = False,
                     isRandWeight = False,
                     weight_bound = [0.6, 1],
                     ):
    if len(clt_data) == 0:
        clt_data = data
    ILs = []
    fail_rate = []
    #filenames = [directory + x for x in os.listdir(directory) if 'Kid' not in x]
    for size in k_range:
        #clustering ...
        print('Processing k = ', size)
        clusters = clustering(clt_data, size)
        #adapt_clusters = [c for i, c in enumerate(clusters) if 'Kid' not in raw_data[i]['imagePath']]
        fakeid_dist_avg, all_distances, information_loss, \
        labelList, fail_prob, df_stats\
        = cluster_gen(latents, clusters, data, raw_data, model,
                      isShown = isShown, rafd_kid = rafd_kid, alpha=alpha, 
                      isAdjustWeight = isAdjustWeight,
                      isBackward = isBackward,
                      isRandWeight = isRandWeight,
                      weight_bound = weight_bound,
                      k = size,
                     )

        sum_IL = sum(information_loss)
        ILs.append(sum_IL)
        fail_rate.append(fail_prob)
    ILs = np.array(ILs)/ len(data)
    return ILs, fail_rate