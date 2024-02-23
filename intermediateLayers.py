
# -*- coding: utf-8 -*-

import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import tensorflow as tf
from keras.models import Model
import cv2


import sys, os, warnings


gpu = sys.argv[ sys.argv.index('-gpu') + 1 ] if '-gpu' in sys.argv else '0'
os.environ['PYTHONHASHSEED'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES']=gpu
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow CUDA load statements
#warnings.filterwarnings('ignore')

from keras import backend as K
import tensorflow as tf

import copy
import argparse
import numpy as np


gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", gpus)

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    tf.config.experimental.set_memory_growth(gpus[int(gpu)], True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import utilArgparse
import utilConst
import utilIO
import util
import CNNmodel

#util.init()

K.set_image_data_format('channels_last')




# ----------------------------------------------------------------------------
def menu():
    parser = argparse.ArgumentParser(description='Data augmentation on test')

    
    parser.add_argument('-db_train_src', required=True, help='Dataset path for training (src imags)')
    parser.add_argument('-db_train_gt', required=True, help='Dataset path for training (gt images)')

    parser.add_argument('-db_test_src', required=False, help='Dataset path to test (src imags)')
    parser.add_argument('-db_test_gt', required=False, help='Dataset path to test (gt images)')

    parser.add_argument('-aug',   nargs='*',
                        choices=utilConst.AUGMENTATION_CHOICES,
                        default=[utilConst.AUGMENTATION_NONE], 
                        help='Data augmentation modes')

    parser.add_argument('-npatches', default=-1, dest='n_pa', type=int,   help='Number of patches to be extracted from training data')
    parser.add_argument('-n_aug', default=0, dest='n_aug', type=int,   help='Number of augmentations applied for each patch')

    parser.add_argument('-n_annotated_patches', default=-1, dest='n_an', type=int,   help='Number of patches to be extracted from training data')

    parser.add_argument('-window_w', default=256, dest='win_w', type=int,   help='width of window')
    parser.add_argument('-window_h', default=256, dest='win_h', type=int,   help='height of window')

    parser.add_argument('-l',          default=4,        dest='n_la',     type=int,   help='Number of layers')
    parser.add_argument('-f',          default=64,      dest='nb_fil',   type=int,   help='Number of filters')
    parser.add_argument('-k',          default=5,        dest='ker',            type=int,   help='kernel size')
    parser.add_argument('-drop',   default=0.2,        dest='drop',          type=float, help='dropout value')
    
    parser.add_argument('-drop_test',   default=None,        dest='drop_test',          type=float, help='dropout value')
    

    parser.add_argument('-pages_train',   default=-1,      type=int,   help='Number of pages to be used for training. -1 to load all the training set.')

    parser.add_argument('-e',           default=200,    dest='ep',            type=int,   help='nb_epoch')
    parser.add_argument('-b',           default=16,     dest='ba',               type=int,   help='batch size')
    parser.add_argument('-verbose',     default=1,                                  type=int,   help='1=show batch increment, other=mute')

    parser.add_argument('--test',   action='store_true', help='Only run test')
    
    
    parser.add_argument('--save',   action='store_true', help='Save resulting images')
    parser.add_argument('-res', required=False, help='File where append the results.')
    parser.add_argument('-gpu',    default='0',    type=str,   help='GPU')

    args = parser.parse_args()

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args

def tpc_result(result):
    return round(result*100,1)
  
def number_to_string(number):
    return str(tpc_result(number)).replace(".",",")


def plot_figures_old(images):
  width = images.shape[0]
  n_filters = images.shape[2]
  plt.figure(figsize=(1.5 * n_filters, 1.5))
  for i in range(n_filters):
    ax = plt.subplot(1,n_filters,i+1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(np.array(images[:,:,i] * 255., dtype=np.uint8), cmap='gray')
  plt.show()

# --------------------------------
def plot_figures(image, images, name_layer):
  width = images.shape[0]
  n_filters = images.shape[2]
  
  plt.figure(figsize=(1.5 * n_filters, 1.5))
  ax = plt.subplot(1,n_filters+1,1)
  ax.set_title(name_layer)
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  
  
  coords = np.where(image==-1)
  image_color = image *255
  for idx in range(len(coords[0])):
    image_color[coords[0][idx], coords[1][idx]] = (0,0,255)
  
  plt.imshow(np.array((image_color), dtype=np.uint8))
  
  for i in range(n_filters):
    ax = plt.subplot(1,n_filters+1,i+2)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = images[:,:,i] 
    coords = np.where(im==-1)
    im = images[:,:,i] * 255.
    im_color = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    for idx in range(len(coords[0])):
      im_color[coords[0][idx], coords[1][idx]] = (0,0,255)
    plt.imshow(np.array(im_color, dtype=np.uint8))
  #plt.show()
  
  
  
def getPredictionsLayer(model, layer_name, input_data):
  model_layers = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
  return model_layers.predict( input_data )


if __name__ == "__main__":
    config = menu()
    print (config)
    
    path_model = utilIO.getPathModel(config)
    utilIO.createParentDirectory(path_model)
    
    input_shape = util.getInputShape(config)
    
    list_src_train = utilIO.listFilesRecursive(config.db_train_src)
    list_gt_train = utilIO.listFilesRecursive(config.db_train_gt)
    assert(len(list_src_train) == len(list_gt_train))

    train_data, val_data = util.create_Validation_and_Training_partitions(
                                        list_src_train=list_src_train, 
                                        list_gt_train=list_gt_train, 
                                        pages_train=config.pages_train)


    print("Training and validation partitioned...")
    print("\tTraining: %d" %(len(train_data)))
    print("\tValidation: %d" %(len(val_data)))

    augmentation_val = ["none"]
    if utilConst.AUGMENTATION_RANDOM in config.aug:
        augmentation_val = ["random"]
    
    model = CNNmodel.get_model(input_shape, config.n_la, config.nb_fil, config.ker, dropout=config.drop, stride=2, dropout_test =config.drop_test)
    
    train_generator = util.create_generator(train_data, config.ba, input_shape, config.n_pa, config.n_an, config.aug)
    val_generator = util.create_generator(val_data, config.ba, input_shape, config.n_pa, config.n_an, augmentation_val)
    
    nb_train_pages = len(train_data)
    nb_val_pages = len(val_data)
    
    epochs = config.ep
    patience = 20
    
    print("Number of effective epochs: " + str(epochs))
    print("Effective patience: " + str(patience))

    if utilConst.AUGMENTATION_RANDOM in config.aug:
        assert(config.n_pa!=-1)
        steps_per_epoch = int(np.ceil((config.n_pa*nb_train_pages)/ config.ba))
    else:
        number_annotated_patches = util.get_number_annotated_patches(train_data, input_shape[0], input_shape[1], config.n_pa)  
        print ("Number of annotated patches: " + str(number_annotated_patches))
        steps_per_epoch = np.ceil(number_annotated_patches/config.ba)

    steps_per_epoch = max(1, steps_per_epoch)
    
    model.summary()
    
    input_data = next(train_generator)
    
    predictions_conv2d =getPredictionsLayer(model, "conv2d", input_data[0])  
    predictions_conv2d_1 =getPredictionsLayer(model, "conv2d_1", input_data[0])    
    predictions_conv2d_2 =getPredictionsLayer(model, "conv2d_2", input_data[0])
    predictions_conv2d_3 =getPredictionsLayer(model, "conv2d_3", input_data[0])
    
    
    
    #print(predictions_conv2d_2.shape)
    plot_figures(input_data[0][6],predictions_conv2d[6], "conv2d")
    plot_figures(input_data[0][6],predictions_conv2d_1[6], "conv2d_1")
    plot_figures(input_data[0][6],predictions_conv2d_2[6], "conv2d_2")
    plot_figures(input_data[0][6],predictions_conv2d_3[6], "conv2d_3")
    
    
    plt.show()