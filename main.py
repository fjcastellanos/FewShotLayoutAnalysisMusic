# -*- coding: utf-8 -*-
from __future__ import print_function
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

    parser.add_argument('-m', required=False, help='Pathfile for the model')
    
    parser.add_argument('-db_train_src', required=True, help='Dataset path for training (src imags)')
    parser.add_argument('-db_train_gt', required=True, help='Dataset path for training (gt images)')

    parser.add_argument('-db_test_src', required=False, help='Dataset path to test (src imags)')
    parser.add_argument('-db_test_gt', required=False, help='Dataset path to test (gt images)')

    parser.add_argument('-aug',   nargs='*',
                        choices=utilConst.AUGMENTATION_CHOICES,
                        default=[utilConst.AUGMENTATION_NONE], 
                        help='Data augmentation modes')

    parser.add_argument('-npatches', default=-1, dest='n_pa', type=int,   help='Number of patches to be extracted from training data')
    
    parser.add_argument('-n_annotated_patches', default=-1, dest='n_an', type=int,   help='Number of patches to be extracted from training data')

    parser.add_argument('-window_w', default=256, dest='win_w', type=int,   help='width of window')
    parser.add_argument('-window_h', default=256, dest='win_h', type=int,   help='height of window')

    parser.add_argument('-l',          default=4,        dest='n_la',     type=int,   help='Number of layers')
    parser.add_argument('-f',          default=64,      dest='nb_fil',   type=int,   help='Number of filters')
    parser.add_argument('-k',          default=5,        dest='ker',            type=int,   help='kernel size')
    parser.add_argument('-drop',   default=0.2,        dest='drop',          type=float, help='dropout value')
    

    parser.add_argument('-pages_train',   default=-1,      type=int,   help='Number of pages to be used for training. -1 to load all the training set.')

    parser.add_argument('-e',           default=200,    dest='ep',            type=int,   help='nb_epoch')
    parser.add_argument('-b',           default=16,     dest='ba',               type=int,   help='batch size')
    parser.add_argument('-verbose',     default=1,                                  type=int,   help='1=show batch increment, other=mute')

    parser.add_argument('--test',   action='store_true', help='Only run test')
    
    
    parser.add_argument('-res', required=False, help='File where append the results.')
    parser.add_argument('-gpu',    default='0',    type=str,   help='GPU')
    parser.add_argument('-no_mask', required=False, action='store_true', help='File where append the results.')
    
    args = parser.parse_args()

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args

def tpc_result(result):
    return round(result*100,1)
  
def number_to_string(number):
    return str(tpc_result(number)).replace(".",",")

if __name__ == "__main__":
    config = menu()
    print (config)
    
    if config.m is None:
      path_model = utilIO.getPathModel(config)
    else:
      path_model = config.m
    utilIO.createParentDirectory(path_model)
    
    input_shape = util.getInputShape(config)
    
    list_src_train = utilIO.listFilesRecursive(config.db_train_src)
    list_gt_train = utilIO.listFilesRecursive(config.db_train_gt)
    assert(len(list_src_train) == len(list_gt_train))

    train_data, val_data = util.create_Validation_and_Training_partitions(
                                        list_src_train=list_src_train, 
                                        list_gt_train=list_gt_train, 
                                        pages_train=config.pages_train)
    
    if config.test == False: # TRAINING MODE

      print("Training and validation partitioned...")
      print("\tTraining: %d" %(len(train_data)))
      print("\tValidation: %d" %(len(val_data)))

      augmentation_val = ["none"]
      if utilConst.AUGMENTATION_RANDOM in config.aug:
        augmentation_val = ["random"]
      
      model = CNNmodel.get_model(input_shape, config.no_mask, config.n_la, config.nb_fil, config.ker, dropout=config.drop, stride=2)
      
      train_generator = util.create_generator(train_data, config.no_mask, config.ba, input_shape, config.n_pa, config.n_an, config.aug)
      val_generator = util.create_generator(val_data, config.no_mask, config.ba, input_shape, config.n_pa, config.n_an, augmentation_val)
      
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
      CNNmodel.train(model, path_model, train_generator, val_generator, steps_per_epoch, nb_val_pages, config.ba, epochs, patience=patience)
    
    else: #TEST MODE
      
      list_src_test = utilIO.listFilesRecursive(config.db_test_src)
      list_gt_test = utilIO.listFilesRecursive(config.db_test_gt)
      assert(len(list_src_test) == len(list_gt_test))
      
      test_data = utilIO.match_SRC_GT_Images(list_src_test, list_gt_test)
      
      print("Obtaining best threshold...(Validation partition)")
      
      threshold=None
      best_fm_val, best_th_val, prec_val, recall_val, dict_predictions = util.compute_best_threshold(path_model, val_data, config.ba, input_shape, nb_annotated_patches=config.n_an, threshold=threshold, with_masked_input=False)
      
      print("Results of the test...")
      with_mask = not config.no_mask
      dict_results = util.test_model(config, path_model, test_data, input_shape, best_th_val, with_mask)
      
      separator = ";"
      print ("SUMMARY:")
      str_result = "VAL"+separator+str(best_th_val) + separator + number_to_string(best_fm_val) + separator + number_to_string(prec_val) + separator + number_to_string(recall_val) + "\n"  #number_to_string(best_fm_val) + separator + number_to_string(prec_val) + separator + number_to_string(recall_val) + separator + str(best_th_val).replace(".", ",") + separator
    
      best_fm_test = dict_results[utilConst.KEY_RESULT][0][0]
      prec_test = dict_results[utilConst.KEY_RESULT][0][1]
      recall_test = dict_results[utilConst.KEY_RESULT][0][2]

      print("Results: " + number_to_string(best_fm_test) + separator + number_to_string(prec_test) + separator + number_to_string(recall_test))
      
      str_result += key + separator + separator + number_to_string(best_fm_test) + separator + number_to_string(prec_test) + separator + number_to_string(recall_test) + separator + "\n"
      print(str_result)
      
      if config.res is not None:
        utilIO.appendString(str_result, config.res, True)
      

      
      