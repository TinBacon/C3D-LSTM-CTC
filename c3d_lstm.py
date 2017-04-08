from __future__ import division

import sys
model_dir='../modelzoo/' # Path to your recipes/modelzoo
sys.path.insert(0,model_dir)

import numpy as np
import theano.tensor as T
import theano

import lasagne
from confusionmatrix import ConfusionMatrix
import os
import uuid

import logging
import argparse

from scipy.misc import imread
import skimage.transform
import pickle

# handle the data
import random
from lasagne.layers import helper
import time
import pdb
import scipy.io as sio
import math

import ctc.py

# in order to do data augmentation
from skimage.transform import SimilarityTransform
from skimage.transform import AffineTransform
from skimage.transform import warp
do_augmentation = False

########################################################################################################################
# argument
np.random.seed(1234)
parser = argparse.ArgumentParser()
# try to go on with the parameters generated last time
parser.add_argument("-lr", type=str, default="0.0005")  # 0.005  0.000005
parser.add_argument("-decayinterval", type=int, default=9)  # 10
parser.add_argument("-decayfac", type=float, default=10)  # 1.5
parser.add_argument("-nodecay", type=int, default=14)  # 15
parser.add_argument("-optimizer", type=str, default='sgd')  # rmsprop
parser.add_argument("-dropout", type=float, default=0.0)
parser.add_argument("-downsample", type=float, default=3.0)
args = parser.parse_args()

########################################################################################################################
# logger 
output_folder = "logs/R3DCNN" + str(uuid.uuid4())[:18].replace('-', '_')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(output_folder, "results.log"), mode='w')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

logger.info('#'*80)
for name, val in sorted(vars(args).items()):
    sep = " "*(35 - len(name))
    logger.info("#{}{}{}".format(name, sep, val))
logger.info('#'*80)

########################################################################################################################
# params init
np.random.seed(123)
TOL = 1e-5
num_batch = 5  # or try 4 # 50  # 100
num_batch_eval = 5  # 50
num_rnn_units = 256
GRAD_CLIP = 100  # for LSTM layer, # All gradients above this will be clipped
num_classes = 83  # 19 15
NUM_EPOCH = 300
LR = float(args.lr)
MONITOR = False   # False
MAX_NORM = 50.0  # 5.0
LOOK_AHEAD = 50

mask_length = 48
clip_length = 16  # 16
clip_stride = 8
H_resize = 240  # 128  # 256
W_resize = 320  # 171  # 342
img_channels = 3

# C3D net
H_net_input = 112  # 224  # 112
W_net_input = 112  # 224  # 112
H_crop = 112  # 224  # 112
W_crop = 112  # 224  # 112

print "mask_length:", mask_length
print "clip_length:", clip_length
clip_init_list = []
for clip_init_idx in range(0, mask_length-clip_length+1, clip_stride):
    clip_end_idx = clip_init_idx + clip_length - 1
    clip_init_list.append(clip_init_idx)

# lstm net
num_steps = len(clip_init_list)    # the length of video clip
print "num_steps:", num_steps


org_drp = args.dropout
sh_drp = theano.shared(lasagne.utils.floatX(args.dropout))

M = T.matrix()
W_ini = lasagne.init.GlorotUniform()
W_ini_gru = lasagne.init.GlorotUniform()
W_proc_ini = lasagne.init.GlorotUniform()
W_class_init = lasagne.init.GlorotUniform()

########################################################################################################################
# input data

Dir_features = "/data/bacon/R3DCNN/"
# Dir_instances = "/shared/RS/"
# file_name_gesture = "gesture_inst"
# file_name_mask = "mask"
# list_masks = os.listdir(Dir_instances + file_name_mask)

# trn, val, tst files
ftrn = open("trn_instance.txt", "r")
fval = open("val_instance.txt", "r")
ftst = open("tst_instance.txt", "r")
list_instances_trn = [line.strip for line in ftrn.readlines()]
list_instances_val = [line.strip for line in fval.readlines()]
list_instances_tst = [line.strip for line in ftst.readlines()]
# list_masks_trn = sio.loadmat('trn_mask.mat')['trn']
# list_masks_val = sio.loadmat('val_mask.mat')['val']
# list_masks_tst = sio.loadmat('tst_mask.mat')['tst']

# sort files by lists
# idx_tag_subj = list_instances[0].find('s')
# idx_tag_label = list_instances[0].find('a')
# for file_instance in list_instances:
#     subject_id = int(file_instance[idx_tag_subj+1: idx_tag_subj+3])

#     if subject_id in split_trn:
#         list_instances_trn.append(file_instance)
#         file_mask = str.replace(file_instance, file_name_gesture, file_name_mask)
#         list_masks_trn.append(file_mask)

#     elif subject_id in split_val:
#         list_instances_val.append(file_instance)
#         file_mask = str.replace(file_instance, file_name_gesture, file_name_mask)
#         list_masks_val.append(file_mask)

#     elif subject_id in split_tst:
#         list_instances_tst.append(file_instance)
#         file_mask = str.replace(file_instance, file_name_gesture, file_name_mask)
#         list_masks_tst.append(file_mask)

from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    from lasagne.layers import dnn
    conv = dnn.Conv2DDNNLayer
    pool = lasagne.layers.MaxPool2DLayer
    reshape = lasagne.layers.ReshapeLayer

elif _platform == "darwin":
    conv = lasagne.layers.Conv2DLayer
    pool = lasagne.layers.MaxPool2DLayer
    reshape = lasagne.layers.ReshapeLayer

# *** the following has not been realized (it has been realized in my_rnn_spn.py):
# if it has generated the mean_image of the training set
# it is better to rewrite step1 as a function

sub_a_image = True
if sub_a_image:
    outfile_mean = Dir_features + "pixel_mean.npy"  # "npz/pixel_mean.npy"
    mean_image = np.load(outfile_mean)
    print 'mean_image.shape', mean_image.shape

sub_3_values = False
if sub_3_values:
    outfile_mean = Dir_features + "img_mean.npy"  # "npz/iamge_mean.npy"
    mean_image = np.load(outfile_mean)
    print 'mean_image.shape', mean_image.shape

########################################################################################################################
# set net
dropout_conv = False
if_flip_filters = False

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer, ReshapeLayer, LSTMLayer, GRULayer
from lasagne.layers.shape import PadLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from lasagne.nonlinearities import softmax
from lasagne.init import Orthogonal, HeNormal, GlorotNormal

net = {}
net['input'] = InputLayer((None, img_channels, clip_length, H_net_input, W_net_input))
net['mask'] = InputLayer((None, num_steps))

# ----------- 1st layer group ---------------
net['conv1a'] = Conv3DDNNLayer(net['input'], 64, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,
                               flip_filters=if_flip_filters,
                               W=lasagne.init.Normal(std=0.01), b=lasagne.init.Constant(0.))
net['pool1']  = MaxPool3DDNNLayer(net['conv1a'],pool_size=(1,2,2),stride=(1,2,2))

# ------------- 2nd layer group --------------
net['conv2a'] = Conv3DDNNLayer(net['pool1'], 128, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,
                               flip_filters=if_flip_filters,
                               W=lasagne.init.Normal(std=0.01), b=lasagne.init.Constant(1.))
net['pool2']  = MaxPool3DDNNLayer(net['conv2a'],pool_size=(2,2,2),stride=(2,2,2))

# ----------------- 3rd layer group --------------
net['conv3a'] = Conv3DDNNLayer(net['pool2'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,
                               flip_filters=if_flip_filters,
                               W=lasagne.init.Normal(std=0.01), b=lasagne.init.Constant(1.))
net['conv3b'] = Conv3DDNNLayer(net['conv3a'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,
                               flip_filters=if_flip_filters,
                               W=lasagne.init.Normal(std=0.01), b=lasagne.init.Constant(1.))
net['pool3']  = MaxPool3DDNNLayer(net['conv3b'],pool_size=(2,2,2),stride=(2,2,2))

# ----------------- 4th layer group --------------
net['conv4a'] = Conv3DDNNLayer(net['pool3'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,
                               flip_filters=if_flip_filters,
                               W=lasagne.init.Normal(std=0.01), b=lasagne.init.Constant(1.))
net['conv4b'] = Conv3DDNNLayer(net['conv4a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,
                               flip_filters=if_flip_filters,
                               W=lasagne.init.Normal(std=0.01), b=lasagne.init.Constant(1.))
net['pool4']  = MaxPool3DDNNLayer(net['conv4b'],pool_size=(2,2,2),stride=(2,2,2))

# ----------------- 5th layer group --------------
net['conv5a'] = Conv3DDNNLayer(net['pool4'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,
                               flip_filters=if_flip_filters,
                               W=lasagne.init.Normal(std=0.01), b=lasagne.init.Constant(1.))
net['conv5b'] = Conv3DDNNLayer(net['conv5a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,
                               flip_filters=if_flip_filters,
                               W=lasagne.init.Normal(std=0.01), b=lasagne.init.Constant(1.))

# We need a padding layer, as C3D only pads on the right, which cannot be done with a theano pooling layer
net['pad'] = PadLayer(net['conv5b'],width=[(0,1),(0,1)], batch_ndim=3)
net['pool5'] = MaxPool3DDNNLayer(net['pad'],pool_size=(2,2,2),pad=(0,0,0),stride=(2,2,2))
net['fc6-1'] = DenseLayer(net['pool5'], num_units=4096, nonlinearity=lasagne.nonlinearities.rectify,
                          W=lasagne.init.Normal(std=0.005), b=lasagne.init.Constant(1.))
net['drop6'] = DropoutLayer(net['fc6-1'], p=0.5)


# if use bidirectional RNN
use_biRNN = False

# replace fc7 with lstm
dim_fc6 = net['drop6'].output_shape
print 'dim_fc6:', dim_fc6
net['fc6_resize'] = ReshapeLayer(net['drop6'], (-1, num_steps) + dim_fc6[1:])
print 'dim_fc6_resize:', net['fc6_resize'].output_shape

if use_biRNN:
    net['lstm7_forward'] = LSTMLayer(net['fc6_resize'], num_units=num_rnn_units, only_return_final=True,
                                     grad_clipping=GRAD_CLIP,
                                     nonlinearity=lasagne.nonlinearities.tanh,
                                     mask_input=net['mask'])
    net['lstm7_backward'] = LSTMLayer(net['fc6_resize'], num_units=num_rnn_units, only_return_final=True, backwards=True,
                                      grad_clipping=GRAD_CLIP,
                                      nonlinearity=lasagne.nonlinearities.tanh,
                                      mask_input=net['mask'])
    net['lstm7'] = lasagne.layers.ConcatLayer([net['lstm7_forward'], net['lstm7_backward']])
else:
    net['lstm7'] = LSTMLayer(net['fc6_resize'], num_units=num_rnn_units, unroll_scan=True, only_return_final=True,
                             grad_clipping=GRAD_CLIP,
                             nonlinearity=lasagne.nonlinearities.tanh,
                             cell_init=Orthogonal(), hid_init=Orthogonal(), learn_init=True,
                             mask_input=net['mask'])

print 'dim_lstm7:', net['lstm7'].output_shape
net['lstm7_dropout'] = DropoutLayer(net['lstm7'], p=0.5)
# l_fc8 = DenseLayer(net['lstm7']_dropout, num_units=num_classes, W=HeNormal(), nonlinearity=softmax)
net['fc8-1'] = DenseLayer(net['lstm7_dropout'], num_units=num_classes, nonlinearity=None,
                        W=lasagne.init.Normal(std=0.01), b=lasagne.init.Constant(0.))
net['prob'] = NonlinearityLayer(net['fc8-1'], softmax)


########################################################################################################################
# build model
# you could also use the model trained by caffe since you can read parameters with pycaffe
used_last_model = False  # if the last model is trained with caffe model, then used_caffe_model should be True
if used_last_model:
    model_last = pickle.load(open(Dir_features + 'c3d_last_model.pkl'))
    print 'the last time trained c3d len(model_last):', len(model_last)
    print 'inherit the parameters from the model trained last time'
    lasagne.layers.set_all_param_values(net['prob'], model_last)

used_pretrained_c3d = True
if used_pretrained_c3d:
    Dir_pretrained_c3d = Dir_features + 'c3d_pretrained.pkl'
    f = open(Dir_pretrained_c3d, 'rb')
    model_pretrained_c3d = pickle.load(f)
    f.close()
    print 'the pretrained c3d model len(model_pretrained_c3d):', len(model_pretrained_c3d)

    # notice that if you add rnn_spn between conv layers, the structure of C3D is also changed
    # by now, since the rnn_spn is add to the last conv layer, the problem can be simplified:
    # lasagne.layers.set_all_param_values(net['conv5b'], model_pretrained_c3d[:-6], trainable=True)
    lasagne.layers.set_all_param_values(net['fc6-1'], model_pretrained_c3d[:-4], trainable=True)

used_caffe_model = True
# Set the weights (takes some time)
# c3d.set_weights(net, 'c3d_model.pkl')
if used_caffe_model and not used_pretrained_c3d and not used_last_model:
    model_file = Dir_features + 'c3d_model.pkl'
    with open(model_file) as f:
        print('Load pretrained weights from %s...' % model_file)
        model = pickle.load(f)
    print('Set the weights...')

    # notice that if you add rnn_spn between conv layers, the structure of C3D is also changed
    # by now, since the rnn_spn is add to the last conv layer, the problem can be simplified:
    # lasagne.layers.set_all_param_values(net['conv5b'], model[:-6], trainable=True)
    lasagne.layers.set_all_param_values(net['fc6-1'], model[:-4], trainable=True)

#########################################################
# try to scale the gradients on the level of parameters like caffe
# by now only change the code with sgd
scale_grad = True
scale_l2_w = False

sym_y = T.imatrix()

# W is regularizable, b is not regularizable (correspondence with caffe)

if scale_grad:
    net['conv1a'].b.tag.grad_scale = 2
    net['conv2a'].b.tag.grad_scale = 2
    net['conv3a'].b.tag.grad_scale = 2
    net['conv3b'].b.tag.grad_scale = 2
    net['conv4a'].b.tag.grad_scale = 2
    net['conv4b'].b.tag.grad_scale = 2
    net['conv5a'].b.tag.grad_scale = 2
    net['conv5b'].b.tag.grad_scale = 2
    net['fc6-1'].b.tag.grad_scale = 2
    # net['fc7-1'].b.tag.grad_scale = 2
    # net['fc8-1'].b.tag.grad_scale = 2
    net['fc8-1'].W.tag.grad_scale = 10
    net['fc8-1'].b.tag.grad_scale = 20

output_train = lasagne.layers.get_output(net['prob'], deterministic=False)
output_eval = lasagne.layers.get_output(net['prob'], deterministic=True)

# compute the cost for training
output_flat = T.reshape(output_train, (-1, num_classes))
#cost = T.nnet.categorical_crossentropy(output_flat+TOL, sym_y.flatten())
cost =  ctc_loss(output_flat+TOL, sym_y.flatten())
cost = T.mean(cost)

# maybe it is necessary to add l2_penalty to the cost
regularizable_params = lasagne.layers.get_all_params(net['prob'], regularizable=True)
print 'the regularizable_params are:'
for p in regularizable_params:
    print p.name

l2_w = 0.0005
all_layers = lasagne.layers.get_all_layers(net['prob'])
l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * l2_w
cost += l2_penalty

# compute the cost for evaluation
output_eval_flat = T.reshape(output_eval, (-1, num_classes))
# cost_eval = T.nnet.categorical_crossentropy(output_eval_flat, sym_y.flatten())
cost_eval =  ctc_loss(output_flat+TOL, sym_y.flatten())
cost_eval = T.mean(cost_eval)

# all_params = lasagne.layers.get_all_params(net['prob'], trainable=True)
trainable_params = lasagne.layers.get_all_params(net['prob'], trainable=True)
print 'the trainable_params are:'
for p in trainable_params:
    print p.name

# all_grads = T.grad(cost, trainable_params)
# all_grads = [T.clip(g, -10, 10) for g in all_grads]  # T.clip(g, -1, 1)
sh_lr = theano.shared(lasagne.utils.floatX(LR))

#########################################################
# try to scale the gradients on the level of parameters like caffe
# by now only change the code with sgd
if scale_grad:
    grads = theano.grad(cost, trainable_params)
    for idx, param in enumerate(trainable_params):
        grad_scale = getattr(trainable_params, 'grad_scale', 1)
        if grad_scale != 1:
            grads[idx] *= grad_scale
# updates = lasagne.updates.momentum(grads, trainable_params, learning_rate=sh_lr, momentum=0.9)

# adam works with lr 0.001
# updates, norm = lasagne.updates.total_norm_constraint(
#     all_grads, max_norm=MAX_NORM, return_norm=True)

if args.optimizer == 'rmsprop':
    updates_opt = lasagne.updates.rmsprop(cost, trainable_params, learning_rate=sh_lr)
    updates = lasagne.updates.apply_momentum(updates_opt, trainable_params, momentum=0.9)

    # updates = lasagne.updates.rmsprop(updates, trainable_params,
    #                                   learning_rate=sh_lr)
elif args.optimizer == 'adam':
    updates_opt = lasagne.updates.adam(cost, trainable_params, learning_rate=sh_lr)
    updates = lasagne.updates.apply_momentum(updates_opt, trainable_params, momentum=0.9)

    # updates = lasagne.updates.adam(updates, trainable_params,
    #                                learning_rate=sh_lr)
elif args.optimizer == 'sgd':
    # Stochastic Gradient Descent (SGD) with momentum
    if scale_grad:
        updates = lasagne.updates.momentum(grads, trainable_params, learning_rate=sh_lr, momentum=0.9)
    else:
        updates = lasagne.updates.momentum(cost, trainable_params, learning_rate=sh_lr, momentum=0.9)

    # updates_opt = lasagne.updates.sgd(cost, trainable_params, learning_rate=sh_lr)
    # updates = lasagne.updates.apply_momentum(updates_opt, trainable_params, momentum=0.9)

    # updates = lasagne.updates.sgd(updates, trainable_params, learning_rate=sh_lr)
    # updates = lasagne.updates.apply_momentum(updates, trainable_params, momentum=0.9)

elif args.optimizer == 'adadelta':
    updates_opt = lasagne.updates.adadelta(cost, trainable_params, learning_rate=sh_lr)
    updates = lasagne.updates.apply_momentum(updates_opt, trainable_params, momentum=0.9)

    # updates = lasagne.updates.adadelta(updates, trainable_params, learning_rate=sh_lr)

elif args.optimizer == 'adagrad':
    updates_opt = lasagne.updates.adagrad(cost, trainable_params, learning_rate=sh_lr)
    updates = lasagne.updates.apply_momentum(updates_opt, trainable_params, momentum=0.9)

    # updates = lasagne.updates.adagrad(updates, trainable_params, learning_rate=sh_lr)

############################################################################################################
# train, val, test
f_train = theano.function([net['input'].input_var, sym_y, net['mask'].input_var], [cost, output_train], updates=updates)
f_eval = theano.function([net['input'].input_var, sym_y, net['mask'].input_var], [cost_eval, output_eval])

num_train = len(list_instances_trn)
num_valid = len(list_instances_val)
num_test = len(list_instances_tst)

batches_train = np.int(math.ceil(num_train / num_batch))
print "batches_train:", batches_train

batches_valid_eval = np.int(math.ceil(num_valid / num_batch_eval))
print 'batches_valid_eval:', batches_valid_eval

batches_test_eval = np.int(math.ceil(num_test / num_batch_eval))
print 'batches_test_eval:', batches_test_eval

best_valid = 0
best_test = 0
look_count = LOOK_AHEAD
# cost_train_lst = []
last_decay = 0

def im_affine_transform(image, scale=1.0, rotation=0, shear=0, translation_y=0, translation_x=0, return_tform=False):
    # the image is already 01c now
    # Assumed image in c01. Convert to 01c for skimage
    image = image.transpose(1, 2, 0)

    # Normalize so that the param acts more like im_rotate, im_translate etc
    scale = 1 / scale
    translation_x = - translation_x
    translation_y = - translation_y

    # shift to center first so that image is rotated around center
    center_shift = np.array((image.shape[0], image.shape[1])) / 2. - 0.5
    tform_center = SimilarityTransform(translation=-center_shift)
    tform_uncenter = SimilarityTransform(translation=center_shift)

    rotation = np.deg2rad(rotation)
    tform = AffineTransform(scale=(scale, scale), rotation=rotation,
                            shear=shear,
                            translation=(translation_x, translation_y))
    tform = tform_center + tform + tform_uncenter

    warped_img = warp(image, tform)

    # Convert back from 01c to c01
    warped_img = warped_img.transpose(2, 0, 1)
    warped_img = warped_img.astype(image.dtype)
    if return_tform:
        return warped_img, tform
    else:
        return warped_img

#########################################################
# epoch
for epoch in range(NUM_EPOCH):
    start_epoch = time.time()

    cost_train_lst = []
    cost_eval_train_lst = []
    cost_eval_valid_lst = []
    cost_eval_test_lst = []

    # eval train
    # shuffle = np.random.permutation(x_train.shape[0])

    # shuffle lines_train
    random.shuffle(list_instances_trn)

    if epoch < 5:
        sh_drp.set_value(lasagne.utils.floatX((epoch)*org_drp/5.0))
    else:
        sh_drp.set_value(lasagne.utils.floatX(org_drp))

    #########################################################
    # train 
    # changed the train to validation
    conf_train = ConfusionMatrix(num_classes)

    # iteration
    for i in range(batches_train):
        start_training_batch = time.time()

        # idx = shuffle[i*num_batch:(i+1)*num_batch]

        instance_init = i * num_batch
        instance_end = min((i + 1) * num_batch, num_train)
        instance_num = instance_end - instance_init

        # read gesture instances based on the list
        y_batch = []
        mask_batch = []
        x_batch = np.empty((instance_num, mask_length, img_channels, H_crop, W_crop), dtype=np.float32)
        idx_clip = 0

        #########################################################
        # batch size
        for line in list_instances_trn[instance_init: instance_end]:

            instance_load = sio.loadmat(line)
            instance = instance_load['gesture_inst']
            # print 'the shape of instance:', instance.shape  # (40, 120, 160, 3)

            # mask_load = sio.loadmat(line.replace('gest', 'mask'))
            # mask = mask_load['mask']
            mask = instance_load['mask']
            mask_batch += [mask.T]

            label = instance_load['gesture_label']
            # label = line[idx_tag_label + 1:idx_tag_label + 3]
            
            # need to have x_batch and y_batch
            y_batch.append(int(label) - 1)

            img = instance[0]
            # print 'the dimension of the raw image:', img.shape  # (120, 160, 3)

            h, w, _ = img.shape
            flip_lr = random.randint(0, 1)
            # flip_lr = 0
            x_clip = np.empty((mask_length, img_channels, H_crop, W_crop), dtype=np.float32)

            # img_resize = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
            if h != H_resize or w != W_resize:
                img_resize = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
            else:
                img_resize = img

            if sub_3_values:
                img_resize = np.array(img_resize, dtype=float) - mean_image

            img_resize = np.rollaxis(img_resize, 2, 0)  # from h*w*3 to 3*h*w
            # print 'the dimension after rolling:', img_resize.shape

            if sub_a_image:
                img_resize = np.array(img_resize, dtype=float)
                img_resize -= mean_image

            # do augmentation
            if do_augmentation:
                # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.
                zoom_range = [1 / 1.2, 1.2]

                rotation_range = [0, 15]  # (0, 360)
                # random rotation [0, 360]
                rotation = np.random.uniform(*rotation_range)  # there is no post-augmentation, so full rotations here!

                log_zoom_range = [np.log(z) for z in zoom_range]
                zoom = np.exp(np.random.uniform(*log_zoom_range))  # for a zoom factor this sampling approach makes more sense.

                # print 'img_resize[0:5] before augmentation:', img_resize[0:5]

                img_resize, tform = im_affine_transform(
                    img_resize, return_tform=True,
                    scale=zoom, rotation=rotation,
                    shear=0,
                    translation_y=0,
                    translation_x=0
                )
                # print 'zoom:', zoom
                # print 'rotation:', rotation
                # print 'after augmentation, img_resize.shape:', img_resize.shape
                # print 'img_resize[0:5] after augmentation:', img_resize[0:5]

            # crop and flip, in python, the index begins from 0
            h_off_max = H_resize - H_crop - 1
            w_off_max = W_resize - W_crop - 1
            h_off = random.randint(0, h_off_max)
            w_off = random.randint(0, w_off_max)
            img_crop = img_resize[:, h_off:h_off + H_crop, w_off:w_off + W_crop]
            #print 'the dimension after cropping:', img_crop.shape

            # flip left-right if chosen
            if flip_lr == 1:
                img_crop = img_crop[:, ::-1]

            # convert to BGR (since inherit the parameters from caffe)
            if used_caffe_model:
                img_crop = img_crop[::-1, :, :]
            
            # subtract the mean
            # img_crop -= MEAN_IMAGE

            x_clip[0, :, :, :] = img_crop

            #########################################################
            #
            for idx_frm in range(1, mask_length):  # [1,2,...,39]
                #print 'go on with idx_frm:', idx_frm

                img = instance[idx_frm]

                # img_resize = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
                if h != H_resize or w != W_resize:
                    img_resize = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
                else:
                    img_resize = img

                if sub_3_values:
                    img_resize = np.array(img_resize, dtype=float) - mean_image

                img_resize = np.rollaxis(img_resize, 2, 0)  # from h*w*3 to 3*h*w
                # print 'the dimension after rolling:', img_resize.shape

                if sub_a_image:
                    img_resize = np.array(img_resize, dtype=float)
                    img_resize -= mean_image

                # do augmentation
                if do_augmentation:
                    img_resize, tform = im_affine_transform(
                        img_resize, return_tform=True,
                        scale=zoom, rotation=rotation,
                        shear=0,
                        translation_y=0,
                        translation_x=0
                    )

                img_crop = img_resize[:, h_off:h_off + H_crop, w_off:w_off + W_crop]

                if flip_lr == 1:
                    img_crop = img_crop[:, ::-1]
                    
                # convert to BGR (since inherit the parameters from caffe)
                if used_caffe_model:
                    img_crop = img_crop[::-1, :, :]

                # subtract the mean
                # img_crop -= MEAN_IMAGE

                x_clip[idx_frm, :, :, :] = img_crop

            # have generated a clip
            x_batch[idx_clip, :, :, :, :] = x_clip
            idx_clip += 1

        #########################################################
        # x_batch and y_batch have been generated
        # from lists to arrays
        y_batch = np.array(y_batch, dtype=np.int32)
        y_batch = np.reshape(y_batch, (y_batch.shape[0], 1))

        # split L into clips
        # you need to take the number of non-zero values of mask into consideration

        x_batch = np.rollaxis(x_batch, 2, 1)  # from n*L*3*h*w to n*3*L*h*w
        x_batch_c3d = np.empty((instance_num, num_steps, img_channels, clip_length, H_crop, W_crop), dtype=np.float32)
        mask_batch_c3d = np.empty((instance_num, num_steps), dtype=np.float32)  # Is L40 too short for L16S8? only 4 time steps
        mask_batch = np.vstack(mask_batch)

        # print x_batch_c3d[:, 1, :].shape  # (5, 3, 16, 112, 112)
        for idx_step, clip_init_idx in enumerate(clip_init_list):
            x_c3d = x_batch[:, :, clip_init_idx:clip_init_idx + clip_length, :]
            # print 'x_c3d.shape:', x_c3d.shape  # (5, 3, 16, 112, 112)
            x_batch_c3d[:, idx_step, :] = x_c3d
            mask_batch_c3d[:, idx_step] = mask_batch[:, clip_init_idx]

        x_batch_c3d_dim = x_batch_c3d.shape
        x_batch_c3d = np.reshape(x_batch_c3d, (x_batch_c3d_dim[0] * x_batch_c3d_dim[1],) + x_batch_c3d_dim[2:])
        # print 'the shape of x_batch_c3d after reshape:', x_batch_c3d.shape

        train_out = f_train(x_batch_c3d, y_batch, mask_batch_c3d)  # y_batch_repeat
        cost_train, _ = train_out[:2]
        # print 'batch_train:', i, 'cost_train:', cost_train

        cost_train_lst += [cost_train]

        # pdb.set_trace()
        cost_eval_train, probs_train = f_eval(x_batch_c3d, y_batch, mask_batch_c3d)
        # print 'batches_train_eval:', i, 'cost_eval_train:', cost_eval_train
        preds_train_flat = probs_train.reshape((-1, num_classes)).argmax(-1)
        conf_train.batch_add(
            y_batch.flatten(),
            preds_train_flat
        )
        cost_eval_train_lst += [cost_eval_train]

    #########################################################
    # pdb.set_trace()
    cost_eval_train_lst = np.array(cost_eval_train_lst)
    cost_eval_train_mean = cost_eval_train_lst.mean()

    cost_train_lst = np.array(cost_train_lst)
    cost_train_mean = cost_train_lst.mean()

    # change learning_rate
    if last_decay > args.decayinterval and epoch > args.nodecay:
        last_decay = 0
        old_lr = sh_lr.get_value(sh_lr)
        new_lr = old_lr / args.decayfac
        sh_lr.set_value(lasagne.utils.floatX(new_lr))
        print "Decay lr from %f to %f" % (float(old_lr), float(new_lr))
    else:
        last_decay += 1
    ########################################################################################################################
    # val
    # changed the test to validation
    conf_valid = ConfusionMatrix(num_classes)

    # iteration
    for i in range(batches_valid_eval):
        start_evaluating_batch = time.time()

        instance_init = i * num_batch_eval
        instance_end = min((i + 1) * num_batch_eval, num_valid)
        instance_num = instance_end - instance_init

        y_batch = []
        mask_batch = []
        x_batch = np.empty((instance_num, mask_length, img_channels, H_crop, W_crop), dtype=np.float32)
        idx_clip = 0
        #########################################################
        # batches size
        for line in list_instances_val[instance_init: instance_end]:

            instance_load = sio.loadmat(line)
            instance = instance_load['gesture_inst']
            # print 'the shape of instance:', instance.shape  # (40, 120, 160, 3)

            # mask_load = sio.loadmat(line.replace('gest', 'mask'))
            # mask = mask_load['mask']
            mask = instance_load['mask']
            mask_batch += [mask.T]

            label = instance_load['gesture_label']
            # label = line[idx_tag_label + 1:idx_tag_label + 3]

            # need to have x_batch and y_batch
            y_batch.append(int(label) - 1)
            
            img = instance[0]
            # print 'the dimension of the raw image:', img.shape

            h, w, _ = img.shape
            # flip_lr = random.randint(0, 1)
            x_clip = np.empty((mask_length, img_channels, H_crop, W_crop), dtype=np.float32)
            
            # need to resize, randomly crop and flip the whole clip
            # img_resize = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
            if h != H_resize or w != W_resize:
                img_resize = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
            else:
                img_resize = img

            if sub_3_values:
                img_resize = np.array(img_resize, dtype=float) - mean_image

            img_resize = np.rollaxis(img_resize, 2, 0)  # from h*w*3 to 3*h*w
            # print 'the dimension after rolling:', img_resize.shape

            if sub_a_image:
                img_resize = np.array(img_resize, dtype=float)
                img_resize -= mean_image

            # only crop the center region
            h_off = np.int((H_resize - H_crop) / 2)
            w_off = np.int((W_resize - W_crop) / 2)

            img_crop = img_resize[:, h_off:h_off + H_crop, w_off:w_off + W_crop]
            # print 'the dimension after cropping:', img_crop.shape

            # # flip left-right if chosen
            # if flip_lr == 1:
            # img_crop = img_crop[:, ::-1]

            # convert to BGR (since inherit the parameters from caffe)
            if used_caffe_model:
                img_crop = img_crop[::-1, :, :]

            # subtract the mean
            # img_crop -= MEAN_IMAGE

            x_clip[0, :, :, :] = img_crop

            #########################################################
            #
            for idx_frm in range(1, mask_length):  # [1,2,...,39]

                img = instance[idx_frm]

                # img_resize = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
                if h != H_resize or w != W_resize:
                    img_resize = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
                else:
                    img_resize = img

                if sub_3_values:
                    img_resize = np.array(img_resize, dtype=float) - mean_image

                img_resize = np.rollaxis(img_resize, 2, 0)  # from h*w*3 to 3*h*w
                # print 'the dimension after rolling:', img_resize.shape

                if sub_a_image:
                    img_resize = np.array(img_resize, dtype=float)
                    img_resize -= mean_image

                img_crop = img_resize[:, h_off:h_off + H_crop, w_off:w_off + W_crop]

                # if flip_lr == 1:
                # img_crop = img_crop[:, ::-1]

                # convert to BGR (since inherit the parameters from caffe)
                if used_caffe_model:
                    img_crop = img_crop[::-1, :, :]

                # subtract the mean
                # img_crop -= MEAN_IMAGE

                x_clip[idx_frm, :, :, :] = img_crop

            # have generated a clip
            x_batch[idx_clip, :, :, :, :] = x_clip
            idx_clip += 1   

        # print 'the number of clips in a batch:', idx_clip
        # print 'len(y_batch)', len(y_batch)

        #########################################################
        # x_batch and y_batch have been generated
        # from lists to arrays
        y_batch = np.array(y_batch, dtype=np.int32)
        y_batch = np.reshape(y_batch, (y_batch.shape[0], 1))

        x_batch = np.rollaxis(x_batch, 2, 1)  # from n*L*3*h*w to n*3*L*h*w
        x_batch_c3d = np.empty((instance_num, num_steps, img_channels, clip_length, H_crop, W_crop), dtype=np.float32)
        mask_batch_c3d = np.empty((instance_num, num_steps),
                                  dtype=np.float32)  # Is L40 too short for L16S8? only 4 time steps
        mask_batch = np.vstack(mask_batch)

        for idx_step, clip_init_idx in enumerate(clip_init_list):
            x_c3d = x_batch[:, :, clip_init_idx:clip_init_idx + clip_length, :]
            x_batch_c3d[:, idx_step, :] = x_c3d
            mask_batch_c3d[:, idx_step] = mask_batch[:, clip_init_idx]

        x_batch_c3d_dim = x_batch_c3d.shape
        x_batch_c3d = np.reshape(x_batch_c3d, (x_batch_c3d_dim[0] * x_batch_c3d_dim[1],) + x_batch_c3d_dim[2:])

        cost_eval_valid, probs_valid = f_eval(x_batch_c3d, y_batch, mask_batch_c3d)

        preds_valid_flat = probs_valid.reshape((-1, num_classes)).argmax(-1)
        conf_valid.batch_add(
            y_batch.flatten(),
            preds_valid_flat
        )

        cost_eval_valid_lst += [cost_eval_valid]

    #########################################################

    cost_eval_valid_lst = np.array(cost_eval_valid_lst)
    cost_eval_valid_mean = cost_eval_valid_lst.mean()

    logger.info(
        "Epoch {} Acc Train = {}, Acc Test = {}".format(
            epoch,
            conf_train.accuracy(),
            conf_valid.accuracy())
    )

    logger.info(
        "Epoch {} cost_train_mean = {}, cost_eval_train_mean = {}, cost_eval_valid_mean = {}".format(
            epoch,
            cost_train_mean,
            cost_eval_train_mean,
            cost_eval_valid_mean)
    )

    if conf_valid.accuracy() > best_valid:
        best_valid = conf_valid.accuracy()
        best_params = helper.get_all_param_values(net['prob'])
        print 'get the current best_params'
        look_count = LOOK_AHEAD

        # save weights
        outfile_model = Dir_features + '/outfile_model/' + "R3DCNN.pkl"
        # print 'type(best_params):', type(best_params)
        # print 'len(best_params):', len(best_params)
        # print best_params
        f = open(outfile_model, 'wb')
        pickle.dump(best_params, f)
        f.close()

    else:
        look_count -= 1

    end_epoch = time.time()
    print 'the whole time of this epoch is:', end_epoch - start_epoch

    if look_count <= 0:
        break