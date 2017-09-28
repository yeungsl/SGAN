import sys
import os
import argparse
import shutil
import numpy as np
import tensorflow as tf
import time
import scipy


''' settings '''
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='logs/test')
parser.add_argument('--save_interval', type = int, default = 1)
parser.add_argument('--num_epoch', type = int, default = 200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--advloss_weight', type=float, default=1.) # weight for adversarial loss
parser.add_argument('--condloss_weight', type=float, default=1.) # weight for conditional loss
parser.add_argument('--entloss_weight', type=float, default=10.) # weight for entropy loss
parser.add_argument('--gen_lr', type=float, default=0.0001) # learning rate for generator
parser.add_argument('--disc_lr', type=float, default=0.0001) # learning rate for discriminator
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()
print(args)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir) # make out_dir if it does not exist, copy current script to out_dir
    print "Created folder {}".format(args.out_dir)
    shutil.copyfile(sys.argv[0], args.out_dir + '/training_script.py')
else:
    print "folder {} already exists. please remove it first.".format(args.out_dir)
    exit(1)

''' specify pre-trained encoder E (a simple CNN)'''
enc_layers = [tf.Varible(features["x"], [-1, 28, 28, 1])]
enc_layers_conv1 = tf.layers.conv2d(inputs=enc_layers[-1], filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
enc_layers.append(enc_layers_conv1)
enc_layers_pool1 = tf.layers.max_pooling2d(inputs=enc_layers[-1], pool_size=[2, 2], strides=2)
enc_layers.append(enc_layers_pool1)
enc_layers_conv2 = tf.layers.conv2d(input=enc_layers[-1], filters=64, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
enc_layers.append(enc_layers_conv2)
enc_layers_pool2 = tf.layers.max_pooling2d(inputs=enc_layers[-1], pool_size=[2,2], strides=2)
enc_layers.append(tf.reshape(enc_layers_pool2, [-1, 7*7*64]))
enc_layers_fc3 = tf.layers.dense(inputs=enc_layers[-1], units=256, activation=tf.nn.relu)
enc_layers.append(enc_layers_fc3)
enc_layers_fc4 = tf.layers.dense(inputs=enc_layers[-1], units=10, activation=tf.nn.softmax)
enc_layers.append(enc_layers_fc4)

''' load pretrained weights for encoder '''
weights_toload = np.load('pretrained/encoder.npz')
weights_list_toload = [weights_toload['arr_{}'.format(k)] for k in range(len(weights_toload.files))]
#LL.set_all_param_values(enc_layers[-1], weights_list_toload)

''' input tensor variable '''
y_1hot = T.matrix()
x = T.tensor4()
lr = T.scalar()
real_fc3 = LL.get_output(enc_layer_fc3)
