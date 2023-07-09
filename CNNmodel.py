
from keras import layers, regularizers, initializers
from keras.models import Model
from keras.layers import Input, Dropout, Activation, MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Masking
from keras.layers import Concatenate, Subtract
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import utilConst
import util



#from configurations import *

tf.random.set_seed(1)  # Fijamos la semilla de TF
np.random.seed(1)  # Fijamos la semilla

# ----------------------------------------------------------------------------

def get_model(input_size, no_mask, nb_layers, nb_filters, k_size, dropout=0.2, stride=1):
    model = __create_network(input_size, no_mask, nb_layers, nb_filters, k_size, dropout, stride)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=0.001,
                        decay_steps=100000,
                        decay_rate=0.99)

    #opt = SGD(lr=0.01)     # unet
    #opt = 'adam'       # adadelta
    #   Defaults:      Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)  # 0.005
    #   Defaults:      Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    #opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.01)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['mse'])

    print(model.summary())

    return model

# ----------------------------------------------------------------------------

def __create_layer_conv(from_layer, nb_filters, k_size, dropout, strides, bn_axis, deconv=False):
    kernel_initializer = initializers.glorot_uniform(seed=42)   # zeros  glorot_uniform  glorot_normal lecun_normal
    kernel_regularizer = regularizers.l2(0.01)  # None 0.01
    activity_regularizer = None  # regularizers.l1(0.01)

    if deconv is not True:
        x = Conv2D( nb_filters, kernel_size=k_size, strides=strides,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer = kernel_regularizer,
                                activity_regularizer = activity_regularizer,
                                padding='same')(from_layer)
    else:
        x = Conv2DTranspose(nb_filters, kernel_size=k_size, strides=strides,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer = kernel_regularizer,
                            activity_regularizer = activity_regularizer,
                            padding='same')(from_layer)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Dropout(dropout, seed=42)(x)

    return x


# ----------------------------------------------------------------------------

# -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->  # https://arxiv.org/pdf/1502.03167.pdf
def __create_network(input_shape, no_mask, nb_layers, nb_filters=32, k_size=3, dropout=0.2, strides=2):
    input_img = Input(input_shape)
    if no_mask == True:
        mask = input_img
    else:
        mask = Masking(mask_value=utilConst.kPIXEL_VALUE_FOR_MASKING)(input_img)

    x = mask
    encoderLayers = [None] * nb_layers
    bn_axis = 1
    if K.image_data_format() == 'channels_last':
        bn_axis = 3

    for i in range(nb_layers):
        x = __create_layer_conv(x, nb_filters, k_size, dropout, strides, bn_axis)
        encoderLayers[i] = x

    encoded = x

    for i in range(nb_layers):
        x = __create_layer_conv(x, nb_filters, k_size, dropout, strides, bn_axis, True)
        ind = nb_layers - i - 2
        if ind >= 0:
            x = layers.add([x, encoderLayers[ind]])

    decoded = Conv2D(1, kernel_size=k_size, strides=1,
                                        kernel_initializer = initializers.glorot_uniform(seed=42),   # 'glorot_uniform', # zeros
                                        kernel_regularizer = None,
                                        activity_regularizer = None,
                                        padding='same', activation='sigmoid')(x)

    return Model(input_img, decoded)


def f_score_metric(y_true, y_pred):
    max_fscore = None

    for i in range(1, 11, 1):
        th = float(i) / 10.0
    
        y_pred_th = K.cast(K.greater(K.clip(y_pred, 0, 1), th), K.floatx())
        true_positives = K.sum((K.clip(y_true * y_pred_th, 0, 1)))
        possible_positives = K.sum((K.clip(y_true, 0, 1)))
        predicted_positives = K.sum((K.clip(y_pred_th, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        if max_fscore is None:
            max_fscore = f1_val
        else:
            max_fscore = K.max(max_fscore, f1_val) 
    return max_fscore

    #y_true_np = y_true.numpy().flatten()
    #y_pred_np = y_pred.numpy().flatten()

    #best_fm, best_th, prec, recall = util.get_best_threshold(y_pred_np, y_true_np, verbose=0, args_th=None)
    #return best_fm

def train(model, path_out_model, train_generator, val_generator, steps_per_epoch, nb_val_pages, batch_size, epochs=200, patience=20):

    steps_per_epoch_val = 10 #int(np.ceil(nb_patches*nb_val_pages / batch_size))

    print("Steps: " + str(steps_per_epoch))
    
    callbacks_list = [
            ModelCheckpoint(
                path_out_model,
                save_best_only=True,
                monitor="val_mse",
                verbose=1,
                mode="min"
            ),
            EarlyStopping(monitor="val_mse", patience=patience, verbose=0, mode="min")
        ]

    model.fit(
            train_generator,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=steps_per_epoch_val,
            callbacks=callbacks_list,
            epochs=epochs
        )