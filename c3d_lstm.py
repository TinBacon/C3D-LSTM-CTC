from __future__ import division

import numpy as np
import theano
import lasagne
import os
import uuid
import logging
import argparse
import skimage.transform
import pickle
import random
from lasagne.layers import helper
import time
import scipy.io as sio
import math

import net_model
from confusionmatrix import ConfusionMatrix

########################################################################################################################
##########
# argument
##########
def argument():

    parser = argparse.ArgumentParser()
    # try to go on with the parameters generated last time
    parser.add_argument("-lr", type=str, default="0.01")        
    parser.add_argument("-decayinterval", type=int, default=9)  
    parser.add_argument("-decayfac", type=float, default=10)    
    parser.add_argument("-nodecay", type=int, default=14)     
    parser.add_argument("-optimizer", type=str, default='sgd') 
    parser.add_argument("-dropout", type=float, default=0.0)
    parser.add_argument("-downsample", type=float, default=3.0)
    
    return parser.parse_args()

########
# logger
########
def log():

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
    
    return logger


########################################################################################################################
#############
# params init
#############
used_caffe_model = True

np.random.seed(123)
args = argument()
org_drp = args.dropout
sh_drp = theano.shared(lasagne.utils.floatX(args.dropout))
logger = log()

# whole net
NUM_EPOCH = 300                     # epoch number
batch_size = 5                      # batch size
batch_size_eval = 5                 # eval batch size
num_classes = 83                    # class number
LOOK_AHEAD = 50                     # 

# c3d net
clip_length = 16                    # c3d clip length
clip_stride = 8                     # c3d clip stride length
H_resize = 120                      # image resize height
W_resize = 160                      # image resize width
img_channels = 3                    # iamge channels
H_crop = 112                        # image crop height
W_crop = 112                        # image crop width

# lstm net
mask_length = 56                    # sequence length

clip_init_list = []                 # every sequence for lstm split into several clips for c3d
for clip_init_idx in range(0, mask_length-clip_length+1, clip_stride):
    clip_init_list.append(clip_init_idx)

steps_num = len(clip_init_list)     # the length of video clip



########################################################################################################################
#############
# input data
#############

Dir_features = "/data/bacon/R3DCNN/"

# read train, valuate, test files list
with open(Dir_features + "lists/trn_clips.txt", "r") as ftrn:
    list_instances_trn = [line.strip() for line in ftrn.readlines()]
with open(Dir_features + "lists/tst_clips.txt", "r") as ftst:
    list_instances_tst = [line.strip() for line in ftst.readlines()]
with open(Dir_features + "lists/val_clips.txt", "r") as fval:
    list_instances_val = [line.strip() for line in fval.readlines()]

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

# load mean file
sub_a_image = True
if sub_a_image:
    outfile_mean = Dir_features + "pixel_mean.npy" 
    mean_image = np.load(outfile_mean)

sub_3_values = False
if sub_3_values:
    outfile_mean = Dir_features + "img_mean.npy"
    mean_image = np.load(outfile_mean)

########################################################################################################################
###############################
# start training and evaluating
###############################
model = net_model.Model(num_classes, clip_length, img_channels, batch_size, batch_size_eval, steps_num)

# train, val, test
f_train, f_eval = model.build_model(Dir_features, args)

num_train = len(list_instances_trn)
num_valid = len(list_instances_val)
num_test = len(list_instances_tst)

batches_train = np.int(math.ceil(num_train / batch_size))
batches_test_eval = np.int(math.ceil(num_test / batch_size_eval))
batches_valid_eval = np.int(math.ceil(num_valid / batch_size_eval))

best_valid = 0
best_test = 0
look_count = LOOK_AHEAD
last_decay = 0

#########################################################
# epoch
for epoch in range(NUM_EPOCH):
    start_epoch = time.time()

    cost_train_lst = []
    cost_eval_train_lst = []
    cost_eval_valid_lst = []
    cost_eval_test_lst = []

    # shuffle lines_train
    random.shuffle(list_instances_trn)

    if epoch < 5:
        sh_drp.set_value(lasagne.utils.floatX((epoch)*org_drp/5.0))
    else:
        sh_drp.set_value(lasagne.utils.floatX(org_drp))

    #########################################################
    #######
    # train 
    #######

    # changed the train to validation 
    conf_train = ConfusionMatrix(num_classes)

    #########################################################
    # iteration
    for i in range(batches_train):
        start_training_batch = time.time()

        instance_init = i * batch_size
        instance_end = min((i + 1) * batch_size, num_train)
        instance_num = instance_end - instance_init

        # read gesture instances based on the list
        y_batch = []
        mask_batch = []
        x_batch = np.empty((instance_num, mask_length, img_channels, H_crop, W_crop), dtype=np.float32)
        idx_clip = 0

        #########################################################
        # batch size
        for line in list_instances_trn[instance_init: instance_end]:

            print("training....  Iteration%d   Batch%d   Data_path%s"%(epoch, i, line))

            instance_load = sio.loadmat(line)
            instance = instance_load['gesture_inst']
            mask = instance_load['mask']
            mask_batch += [mask]
            label = instance_load['gesture_label']
            
            # need to have x_batch and y_batch
            y_batch.append(int(label) - 1)
            x_clip = np.empty((mask_length, img_channels, H_crop, W_crop), dtype=np.float32)
            flip_lr = random.randint(0, 1)

            img = instance[0]
            h, w, _ = img.shape
            
            ###########################
            # resize and crop the image
            ###########################
            if h != H_resize or w != W_resize:
                img_resize = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
            else:
                img_resize = img

            if sub_3_values:
                img_resize = np.array(img_resize, dtype=float) - mean_image

            # from h*w*3 to 3*h*w
            img_resize = np.rollaxis(img_resize, 2, 0)  

            if sub_a_image:
                img_resize = np.array(img_resize, dtype=float)
                img_resize -= mean_image

            # crop and flip, in python, the index begins from 0
            h_off_max = H_resize - H_crop - 1
            w_off_max = W_resize - W_crop - 1
            h_off = random.randint(0, h_off_max)
            w_off = random.randint(0, w_off_max)
            img_crop = img_resize[:, h_off:h_off + H_crop, w_off:w_off + W_crop]

            # flip left-right if chosen
            if flip_lr == 1:
                img_crop = img_crop[:, ::-1]

            # convert to BGR (since inherit the parameters from caffe)
            if used_caffe_model:
                img_crop = img_crop[::-1, :, :]

            x_clip[0, :, :, :] = img_crop

            #########################################################
            # sequence size
            for idx_frm in range(1, mask_length): 

                img = instance[idx_frm]

                ###########################
                # resize and crop the image
                ###########################
                if h != H_resize or w != W_resize:
                    img_resize = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
                else:
                    img_resize = img

                if sub_3_values:
                    img_resize = np.array(img_resize, dtype=float) - mean_image

                # from h*w*3 to 3*h*w
                img_resize = np.rollaxis(img_resize, 2, 0)  

                if sub_a_image:
                    img_resize = np.array(img_resize, dtype=float)
                    img_resize -= mean_image

                img_crop = img_resize[:, h_off:h_off + H_crop, w_off:w_off + W_crop]

                if flip_lr == 1:
                    img_crop = img_crop[:, ::-1]
                    
                # convert to BGR (since inherit the parameters from caffe)
                if used_caffe_model:
                    img_crop = img_crop[::-1, :, :]

                x_clip[idx_frm, :, :, :] = img_crop

            # have generated a clip
            x_batch[idx_clip, :, :, :, :] = x_clip
            idx_clip += 1

        #########################################################
        # put iteration data into net

        # x_batch, y_batch and mask_batch have been generated
        x_batch = np.rollaxis(x_batch, 2, 1)    # from n*L*3*h*w to n*3*L*h*w

        y_batch = np.array(y_batch, dtype=np.int32)
        y_batch = np.reshape(y_batch, (y_batch.shape[0], 1))
        
        mask_batch = np.vstack(mask_batch)

        # transfer x and mask batch into batch with c3d
        x_batch_c3d = np.empty((instance_num, steps_num, img_channels, clip_length, H_crop, W_crop), dtype=np.float32)
        mask_batch_c3d = np.empty((instance_num, steps_num), dtype=np.float32) 
        for idx_step, clip_init_idx in enumerate(clip_init_list):

            x_c3d = x_batch[:, :, clip_init_idx : clip_init_idx+clip_length, :]
            x_batch_c3d[:, idx_step, :] = x_c3d

            mask_c3d = mask_batch[:, clip_init_idx]
            mask_batch_c3d[:, idx_step] = mask_c3d

        x_batch_c3d_dim = x_batch_c3d.shape
        x_batch_c3d = np.reshape(x_batch_c3d, (x_batch_c3d_dim[0] * x_batch_c3d_dim[1],) + x_batch_c3d_dim[2:])

        # train a batch and get results
        train_out = f_train(x_batch_c3d, y_batch, mask_batch_c3d)
        cost_train, _ = train_out[:2]
        cost_train_lst += [cost_train]

        # eval a batch and get results
        cost_eval_train, probs_train = f_eval(x_batch_c3d, y_batch, mask_batch_c3d)
        preds_train_flat = probs_train.reshape((-1, num_classes)).argmax(-1)
        conf_train.batch_add(y_batch.flatten(), preds_train_flat)
        cost_eval_train_lst += [cost_eval_train]

    #########################################################
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
        print("Decay lr from %f to %f"%(float(old_lr), float(new_lr)))
    else:
        last_decay += 1

    #########################################################
    #####
    # val
    #####

    # changed the test to validation
    conf_valid = ConfusionMatrix(num_classes)

    #########################################################
    # iteration
    for i in range(batches_valid_eval):
        start_evaluating_batch = time.time()

        instance_init = i * batch_size_eval
        instance_end = min((i + 1) * batch_size_eval, num_valid)
        instance_num = instance_end - instance_init
        
        y_batch = []
        mask_batch = []
        x_batch = np.empty((instance_num, mask_length, img_channels, H_crop, W_crop), dtype=np.float32)
        idx_clip = 0
        #########################################################
        # batches size
        for line in list_instances_val[instance_init: instance_end]:

            print("evaluating....  Iteration%d   Batch%d   Data_path%s"%(epoch, i, line))

            instance_load = sio.loadmat(line)
            instance = instance_load['gesture_inst']
            mask = instance_load['mask']
            mask_batch += [mask.T]
            label = instance_load['gesture_label']

            # need to have x_batch and y_batch
            y_batch.append(int(label) - 1)
            x_clip = np.empty((mask_length, img_channels, H_crop, W_crop), dtype=np.float32)

            img = instance[0]
            h, w, _ = img.shape

            ###########################
            # resize and crop the image
            ###########################    
            # need to resize, randomly crop and flip the whole clip
            if h != H_resize or w != W_resize:
                img_resize = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
            else:
                img_resize = img

            if sub_3_values:
                img_resize = np.array(img_resize, dtype=float) - mean_image

            img_resize = np.rollaxis(img_resize, 2, 0)  # from h*w*3 to 3*h*w

            if sub_a_image:
                img_resize = np.array(img_resize, dtype=float)
                img_resize -= mean_image

            # only crop the center region
            h_off = np.int((H_resize - H_crop) / 2)
            w_off = np.int((W_resize - W_crop) / 2)

            img_crop = img_resize[:, h_off:h_off + H_crop, w_off:w_off + W_crop]

            # convert to BGR (since inherit the parameters from caffe)
            if used_caffe_model:
                img_crop = img_crop[::-1, :, :]

            x_clip[0, :, :, :] = img_crop

            #########################################################
            # sequence size
            for idx_frm in range(1, mask_length):

                img = instance[idx_frm]

                if h != H_resize or w != W_resize:
                    img_resize = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
                else:
                    img_resize = img

                if sub_3_values:
                    img_resize = np.array(img_resize, dtype=float) - mean_image

                img_resize = np.rollaxis(img_resize, 2, 0)  # from h*w*3 to 3*h*w

                if sub_a_image:
                    img_resize = np.array(img_resize, dtype=float)
                    img_resize -= mean_image

                img_crop = img_resize[:, h_off:h_off + H_crop, w_off:w_off + W_crop]

                # convert to BGR (since inherit the parameters from caffe)
                if used_caffe_model:
                    img_crop = img_crop[::-1, :, :]

                x_clip[idx_frm, :, :, :] = img_crop

            # have generated a clip
            x_batch[idx_clip, :, :, :, :] = x_clip
            idx_clip += 1   
        
        #########################################################
        # put iteration data into net

        # x_batch, y_batch and mask_batch have been generated
        x_batch = np.rollaxis(x_batch, 2, 1)  # from n*L*3*h*w to n*3*L*h*w
        
        y_batch = np.array(y_batch, dtype=np.int32)
        y_batch = np.reshape(y_batch, (y_batch.shape[0], 1))

        mask_batch = np.vstack(mask_batch).T
        
        # transfer x and mask batch into batch with c3d
        mask_batch_c3d = np.empty((instance_num, steps_num), dtype=np.float32) 
        x_batch_c3d = np.empty((instance_num, steps_num, img_channels, clip_length, H_crop, W_crop), dtype=np.float32)
        for idx_step, clip_init_idx in enumerate(clip_init_list):

            x_c3d = x_batch[:, :, clip_init_idx:clip_init_idx + clip_length, :]
            x_batch_c3d[:, idx_step, :] = x_c3d

            batch_c3d = mask_batch[:, clip_init_idx]
            mask_batch_c3d[:, idx_step] = batch_c3d

        x_batch_c3d_dim = x_batch_c3d.shape
        x_batch_c3d = np.reshape(x_batch_c3d, (x_batch_c3d_dim[0] * x_batch_c3d_dim[1],) + x_batch_c3d_dim[2:])

        # evaluate a batch and get results
        cost_eval_valid, probs_valid = f_eval(x_batch_c3d, y_batch, mask_batch_c3d)

        preds_valid_flat = probs_valid.reshape((-1, num_classes)).argmax(-1)
        conf_valid.batch_add(y_batch.flatten(), preds_valid_flat)

        cost_eval_valid_lst += [cost_eval_valid]

    #########################################################
    ########
    # logger
    ########
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
        best_params = helper.get_all_param_values(model.net['prob'])
        print('get the current best_params')
        look_count = LOOK_AHEAD

        # save weights
        outfile_model = Dir_features + '/outfile_model/' + "R3DCNN.pkl"
        f = open(outfile_model, 'wb')
        pickle.dump(best_params, f)
        f.close()
    else:
        look_count -= 1
    
    #########################################################

    end_epoch = time.time()
    print('the whole time of this epoch is:', end_epoch - start_epoch)

    if look_count <= 0:
        break