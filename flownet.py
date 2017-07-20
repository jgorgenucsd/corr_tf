import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.engine.topology import Layer
from keras.models import Model, Sequential
from keras import activations
from keras.layers import Activation, Input, Reshape, merge, Lambda, Dropout, Flatten, Dense,LSTM
from keras.layers.merge import add,concatenate,dot
from keras.layers.convolutional import Convolution2D, Deconvolution2D, ZeroPadding2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add,Concatenate,Dot
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD
import itertools
#from keras.utils.visualize_util import plot
import random
from scipy import misc
from scipy.linalg import logm, expm
import pandas as pd
import scipy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, \
    img_to_array, load_img
from os import listdir
from os.path import isfile, join
import matplotlib
HEADLESS = False
if HEADLESS:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Use the custom correlation layer or build one from tensorflow slice operations
use_custom_correlation = True
if use_custom_correlation:
  import correlation_layer as cl

QUICK_DEBUG = True
BATCH_SIZE = 3
num_epochs = 10
num_train_sets = 9
loss_order = 2
batch_size = BATCH_SIZE



def myDot():
    return Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0],x[1]),axis=-1,keep_dims=True),name = 'myDot')

def get_padded_stride(b,displacement_x,displacement_y,height_8=384/8,width_8=512/8):
    slice_height = height_8- abs(displacement_y)
    slice_width = width_8 - abs(displacement_x)
    start_y = abs(displacement_y) if displacement_y < 0 else 0
    start_x = abs(displacement_x) if displacement_x < 0 else 0
    top_pad    = displacement_y if (displacement_y>0) else 0
    bottom_pad = start_y
    left_pad   = displacement_x if (displacement_x>0) else 0
    right_pad  = start_x
    
    gather_layer = Lambda(lambda x: tf.pad(tf.slice(x,begin=[0,start_y,start_x,0],size=[-1,slice_height,slice_width,-1]),paddings=[[0,0],[top_pad,bottom_pad],[left_pad,right_pad],[0,0]]),name='gather_{}_{}'.format(displacement_x,displacement_y))(b)
    return gather_layer

def get_correlation_layer(conv3_pool_l,conv3_pool_r,max_displacement=20,stride2=2,height_8=384/8,width_8=512/8):
    layer_list = []
    dotLayer = myDot()
    for i in range(-max_displacement, max_displacement+stride2,stride2):
        for j in range(-max_displacement, max_displacement+stride2,stride2):
            slice_b = get_padded_stride(conv3_pool_r,i,j,height_8,width_8)
            current_layer = dotLayer([conv3_pool_l,slice_b])
            layer_list.append(current_layer)
    return Lambda(lambda x: tf.concat(x, 3),name='441_output_concatenation')(layer_list)
    

def getEncoderModel(height = 384, width = 512,batch_size=32):
    print "Generating model with height={}, width={},batch_size={}".format(height,width,batch_size)

    ## convolution model
    conv_activation = lambda x: activations.relu(x,alpha=0.1) # Use the activation from the FlowNetC Caffe implementation
    #conv_activation = "elu"
    # left and model
    input_l = Input(batch_shape=(batch_size,height, width, 3), name='pre_input')
    input_r = Input(batch_shape=(batch_size,height, width, 3), name='nxt_input')
    #layer 1, output of layer 1 is height/2 x width/2
    conv1 = Convolution2D(64,(7,7), strides = 2,batch_size=batch_size, padding = 'same', name = 'conv1',activation=conv_activation)
    conv1_l = conv1(input_l)
    conv1_r = conv1(input_r)

    #layer 2 output of layer 2 is height/4 x width/4
    conv2 = Convolution2D(128, (5, 5), strides = 2,padding = 'same', name='conv2',activation=conv_activation)
    conv2_l = conv2(conv1_l)
    conv2_r = conv2(conv1_r)

    #layer 3 output of layer 3 is height/8 x width8
    conv3 = Convolution2D(256, (5, 5), strides = 2,padding = 'same', name='conv3',activation=conv_activation)
    conv3_l = conv3(conv2_l)
    conv3_r = conv3(conv2_r)


    # merge
    print "Generating Correlation layer..."
    if use_custom_correlation:
      corr_layer = Lambda( lambda x: cl.corr(a=x[0],b=x[1],stride=2,max_displacement=20), name= "correlation_layer")([conv3_l,conv3_r])
    else:
      corr_layer = get_correlation_layer(conv3_l, conv3_r,max_displacement=20,stride2=2,height_8=height/8,width_8=width/8)
    # merged convolution
    conv3_l_redir = Convolution2D(32,(1,1),name="conv_redir",activation=conv_activation)(conv3_l)
    conv3_l_with_corr = concatenate([conv3_l_redir,corr_layer],name="concatenated_correlation")
    conv3_1 = Convolution2D(256, (3, 3), padding = 'same', name='conv3_1',activation=conv_activation)(conv3_l_with_corr)
    
    #layer 4, output of layer 4 is height/16 x width/16
    conv4 = Convolution2D(512, (3, 3), strides=2,padding = 'same', name='conv4',activation=conv_activation)(conv3_1)
    height_16 = height/16; width_16 = width/16
    conv4_1 = Convolution2D(512, (3, 3), padding = 'same', name='conv4_1',activation=conv_activation)(conv4)
    
    # layer 5, now /32
    conv5 = Convolution2D(512, (3, 3), strides = 2, padding = 'same', name='conv5',activation=conv_activation)(conv4_1)
    height_32 = height_16/2; width_32 = width_16/2
    conv5_1 = Convolution2D(512, (3, 3), padding = 'same', name='conv5_1',activation=conv_activation)(conv5)
    
    # Layer 6, now /64
    conv6 = Convolution2D(1024, (3, 3), strides= 2,padding = 'same', name='conv6',activation=conv_activation)(conv5_1)
    height_64 = height_32/2; width_64 = width_32/2

    print "Compiling..."

    optimizer = SGD(nesterov=True, lr=0.00001, momentum=0.1,decay=0.001);
    model = Model(inputs = [input_l, input_r], outputs = conv6)
    model.compile(optimizer=optimizer,loss='mean_squared_error')
    print "Done"

    return model


if __name__ == '__main__':
    height = 384
    width = 512
    encoderModel = getEncoderModel(height=height, width=width,batch_size = batch_size);
    encoderModel.summary()

