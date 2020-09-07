import imutils
import dlib
import cv2
import face_recognition_models
from imutils import face_utils

import menpo.io as mio
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_68_trimesh
from menpofit.aam import HolisticAAM, LinearAAM
# from menpofit.aam import PatchAAM
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional,AlternatingInverseCompositional
#from menpofit.sdm import SupervisedDescentFitter
from menpofit.fitter import noisy_shape_from_bounding_box
# from menpodetect import load_dlib_frontal_face_detector
import pickle
from tqdm import tqdm
from menpo.feature import fast_dsift, dsift, igo, no_op, double_igo
import imutils.paths
import numpy as np
import matplotlib.pyplot as plt
import partitioning
import utils
import evaluation
import PIL
import cv2

def process(image, crop_proportion=0.2, max_diagonal=400):
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    labeller(image, 'PTS', face_ibug_68_to_face_ibug_68_trimesh)
    return image

def train_aam(path_to_images):
    training_images = mio.import_images(path_to_images, verbose=True)
    training_images = training_images.map(process)
    aam = HolisticAAM(training_images, 
                        group='PTS', 
                        diagonal=150,
                      scales=(0.5, 1.0), verbose=True,
                      #holistic_features=double_igo,
                      max_shape_components=40, max_appearance_components=300
                     )
    fitter = LucasKanadeAAMFitter(aam, 
                                  lk_algorithm_cls=AlternatingInverseCompositional,
                                  n_shape=[10, 40], n_appearance=[60, 300]
                                 )
    return aam, fitter

#RGB version
def aam_features_rgb(fitter, img_dir='', isShown=False, save_path=''):
    image_paths = list(imutils.paths.list_images(img_dir))
    image_paths.sort()
    images = mio.import_images(img_dir)
    result = []
    for image, image_path in tqdm(zip(images, image_paths), total=len(image_paths)):
        aam_feat = {'shape': {}, 'texture': {}}
        # image.view()
        for channel, color in enumerate(['R', 'G', 'B']):
            # print(channel)
            grey_image = image.as_greyscale(mode='channel', channel=channel)
            # mage.view()
            fitting_result = fitting_img(grey_image, fitter)
            if isShown == True:
                fitting_result.view()
                plt.show()
            shape_param = {color: list(fitting_result.shape_parameters[-1])}
            # print(shape_param)
            aam_feat['shape'].update(shape_param)
            appearance_param = {color: list(
                fitting_result.appearance_parameters[-1])}
            aam_feat['texture'].update(appearance_param)
        aam_feat['shape'].update(dict.fromkeys(['R','G', 'B'], list( np.mean( list(aam_feat['shape'].values()), axis=0 ) ) ))
        d = [{"imagePath": image_path,
              "encoding": flattening_aam_feat(aam_feat),
              "shape": aam_feat['shape'],
              "texture": aam_feat['texture']
              }]
        result.extend(d)
    if save_path != '':
        print("[INFO] serializing encodings...")
        f = open(save_path, "wb")
        f.write(pickle.dumps(result))
        f.close()

#[NOTE] non RGB version
def aam_features_gray(fitter, img_dir='', isShown=False, save_path=''):
    image_paths = list(imutils.paths.list_images(img_dir))
    image_paths.sort()
    images = mio.import_images(img_dir)
    result = []
    for image, image_path in tqdm(zip(images, image_paths), total=len(image_paths)):
        aam_feat = {'shape': {}, 'texture': {}}
        # image.view()
        # print(channel)
        image.view_landmarks()
        plt.show()
        grey_image = image.as_greyscale(mode='luminosity')
        # mage.view()
        fitting_result = fitting_img(grey_image, fitter)
        if isShown==True:
            fitting_result.view()
            plt.show()
            recon_img = recon_img_from_fitting_result(fitting_result, fitter.aam)
            recon_img.view()
            plt.show()
        shape_param = list(fitting_result.shape_parameters[-1])
        # print(shape_param)
        aam_feat['shape'] = shape_param

        appearance_param = list(fitting_result.appearance_parameters[-1])
        aam_feat['texture'] = appearance_param

        d = [{"imagePath": image_path,
              "encoding": aam_feat['shape'] + aam_feat['texture'],
              "shape": aam_feat['shape'],
              "texture": aam_feat['texture']
              }]
        result.extend(d)
    if save_path != '':
        print("[INFO] serializing encodings...")
        f = open(save_path, "wb")
        f.write(pickle.dumps(result))
        f.close()

def flattening_aam_feat(aam_feat):
    shape_flatten = sum(aam_feat['shape'].values(), [])
    texture_flatten = sum(aam_feat['texture'].values(), [])
    encode = shape_flatten + texture_flatten
    return encode

#[NOTE] Only use the load_dlib_frontal_face_detector for the first time when there is no landmark for test images
# This one works very bad
def fitting_img(image, fitter):
    # detect = load_dlib_frontal_face_detector()
    # bboxes = detect(image)
    # initial_bbox = bboxes[0]
    # fitting_result = fitter.fit_from_bb(image, initial_bbox,
    #                                     #max_iters=[15, 5]
    #                                     )
    # obtain groubnd truth (original) landmarks
    gt_s = image.landmarks['PTS']
    # generate initialization landmarks
    initial_s = noisy_shape_from_bounding_box(gt_s, gt_s.bounding_box())
    
    fitting_result = fitter.fit_from_shape(image, initial_s,max_iters=20, gt_shape=gt_s, return_costs=True)

    return fitting_result

def reconstruct_gray_img(image, aam, fitter):
    fitting_result = fitting_img(image, fitter)
    scale_index = -1
    n_shape_components = aam.shape_models[scale_index].model.n_components

    shape_weights=fitting_result.shape_parameters[-1][n_shape_components:] 
    appearance_weights= fitting_result.appearance_parameters[-1]

    sm = aam.shape_models[scale_index].model
    am = aam.appearance_models[scale_index]

    shape_instance = sm.instance(shape_weights)
    appearance_instance = am.instance(appearance_weights)
    recon_img = aam._instance(scale_index, 
                            shape_instance, 
                            appearance_instance)
    return recon_img
    
def recon_img_from_fitting_result(fitting_result, aam):
    scale_index = -1
    n_shape_components = aam.shape_models[scale_index].model.n_components

    shape_weights=fitting_result.shape_parameters[-1][n_shape_components:] 
    appearance_weights= fitting_result.appearance_parameters[-1]

    sm = aam.shape_models[scale_index].model
    am = aam.appearance_models[scale_index]

    shape_instance = sm.instance(shape_weights)
    appearance_instance = am.instance(appearance_weights)
    #print('Change!')
    recon_img = aam._instance(scale_index, 
                            shape_instance, 
                            appearance_instance)
    # print('fitting_result: ', aam)
    # recon_img = aam.appearance_reconstructions([ fitting_result.appearance_parameters[-1] ], fitting_result.n_iters_per_scale)
    return recon_img

def aam_re_generator(image, aam, fitter):
    color_channels = [image.as_greyscale(mode='channel', channel=i) for i in range(0,3)]
    recon_channels = []
    # box = box_extract(image)
    # print(box)
    for i, c in enumerate(color_channels):
        recon_channel = reconstruct_gray_img(c, aam, fitter)
        if i == 0:
            shape = recon_channel.shape
        recon_channel=recon_channel.resize( shape )
        recon_channel = recon_channel.rescale_pixels(0,1).as_imageio()
        recon_channels.append(recon_channel)
    w,h = recon_channels[0].shape
    rgbArray = np.zeros((w,h,3), 'uint8')
    for i in range(0,3):
        rgbArray[..., i] = recon_channels[i]
    color_img = PIL.Image.fromarray(rgbArray)
    return color_img

def img_from_params(shape_weights, appearance_weights, aam):
    scale_index = -1
    n_shape_components = aam.shape_models[scale_index].model.n_components
    sm = aam.shape_models[scale_index].model
    am = aam.appearance_models[scale_index]

    shape_instance = sm.instance(shape_weights[n_shape_components:])
    appearance_instance = am.instance(appearance_weights)
    
    recon_img = aam._instance(scale_index, shape_instance, appearance_instance)
    #recon_img.view()
    return recon_img

def aam_gen_from_params(shape_params, appearance_params, aam):
    recon_channels = {}
    #print(appearance_params.keys())
    for key in appearance_params.keys():
        #print(key)
        recon_channels[key] = img_from_params(shape_params[key], 
                                             appearance_params[key], 
                                             aam)
        if key == list(appearance_params.keys())[0]:
            shape = recon_channels[key].shape
        recon_channels[key]=recon_channels[key].resize( shape )
        recon_channels[key] = recon_channels[key].rescale_pixels(0,1).as_imageio()
    w,h = recon_channels['R'].shape
    rgbArray = np.zeros((w,h,3), 'uint8')
    for i, color in zip(range(0,3), ['R', 'G', 'B']):
        rgbArray[..., i] = recon_channels[color]
    color_img = PIL.Image.fromarray(rgbArray)
    return color_img

# #non RGB version
# def aam_gen_from_params(shape_params, appearance_params, aam):
#     #print(appearance_params.keys())
#     #print(key)
#     recon_channel = img_from_params(shape_params, 
#                                          appearance_params, 
#                                          aam).rescale_pixels(0,1).as_imageio()
#     return PIL.Image.fromarray(recon_channel, mode='L')

#RGB version

def avg_aam_feat(aam_encodes):
    shapes = [d['shape'] for d in aam_encodes]
    textures = [d['texture'] for d in aam_encodes]
    encodings = [d['encoding'] for d in aam_encodes]
    img_paths = [d['imagePath'] for d in aam_encodes]
    result = {"imagePath": img_paths,
              "encoding": np.mean(encodings, axis=0),
              "shape": {},
              "texture": {}
              }
    for key in textures[0].keys():
        all_shape = []
        result['shape'][key] = np.mean([d[key] for d in shapes], axis=0)
        result['texture'][key] = np.mean([d[key] for d in textures], axis=0)
    return result

#non RGB version
# def avg_aam_feat(aam_encodes):
#     shapes = [d['shape'] for d in aam_encodes]
#     textures = [d['texture'] for d in aam_encodes]
#     encodings = [d['encoding'] for d in aam_encodes]
#     img_paths = [d['imagePath'] for d in aam_encodes]
#     result = {"imagePath": img_paths,
#               "encoding": np.mean(encodings, axis=0),
#               "shape": [],
#               "texture": []
#               }
#     result['shape'] = np.mean([d for d in shapes], axis=0)
#     result['texture'] = np.mean([d for d in textures], axis=0)
#     return result

def pil_to_cv_img(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def evaluate_aam(k_range, data, ref_data, aam, img_dir, clustering=partitioning.kNN_partition, isShown=False):
    information_loss = []
    fail_prob = []
    encodes = [d['encoding'] for d in data]
    for k in tqdm(k_range, position=0, desc='[k values]'):
        clusters = clustering(encodes, cluster_size=k, method='random')
        df_stats = utils.binding_encode_cltId(ref_data, clusters)
        loss = []
        for cluster in set(clusters):
        #for cluster in tqdm(set(clusters)):
            mem_ids = utils.get_mems_of_cluster(cluster, clusters)
            mem_names = [data[idx]['imagePath'].split('/')[-1] for idx in mem_ids]
            aam_encodes = [data[idx] for idx in mem_ids ]
            avg_encodes = avg_aam_feat(aam_encodes)
            avg_img = aam_gen_from_params(avg_encodes['shape'], 
                                          avg_encodes['texture'],
                                          aam
                                         )
            if isShown:
                plt.imshow(avg_img)
                plt.show()
            avg_img = pil_to_cv_img(avg_img)
            # print('This cluster: ', cluster)
            # print('All clusters: ', clusters)
            # if isShown:
            #     plt.imshow(avg_img, cmap='gray')
            #     plt.show()
            quality_fakeid,_, df_stats = evaluation.dist_to_fakeid(avg_img, 
                                                                 cluster, 
                                                                 clusters, 
                                                                 ref_data, 
                                                                 df_stats)
            loss.append(quality_fakeid)
        df_stats['disclosure'] = df_stats['fakeid_dist'].map(lambda row: 1 if row > 0.6 else 0)
        disclosure_prob = len(df_stats[df_stats['disclosure'] == 0]) / len(clusters)
        fail_prob.append(disclosure_prob)
        #print(fail_prob)
        information_loss.append(sum(loss)/len(clusters))
        #print(information_loss)
    return information_loss, fail_prob

def dlib_landmark(img_path, img_dir):
    detector = dlib.get_frontal_face_detector()
    predictor_68_point_model = face_recognition_models.pose_predictor_model_location()

    predictor = dlib.shape_predictor(predictor_68_point_model)
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img_path)
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    shapes = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            # show the output image with the face detections + facial landmarks
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        shape = swap_np_columns(shape)
        shapes.append(shape)
        if i == 0:
            pts_file = img_dir + img_path.split('/')[-1].split('.')[0] + '.pts'
            print(pts_file)
            mio.export_landmark_file(menpo.shape.PointCloud(shape),
                                     pts_file,
                                     overwrite=True)
            mio_img = mio.import_image(img_path)
            mio_img.view_landmarks()
            plt.show()
            
    return shapes

def swap_np_columns(array):
    my_array = array.copy()
    my_array[:, 0], my_array[:, 1] = my_array[:, 1], my_array[:, 0].copy()
    return my_array