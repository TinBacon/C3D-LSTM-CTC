from __future__ import division
import numpy as np
from scipy.misc import imread, imsave, imresize
import skimage.transform
import pickle
import pdb
import scipy.io as sio
import cv2

H_resize = 120  # 128
W_resize = 160  # 171
img_channels = 3

file_train = "/data/bacon/R3DCNN/lists/trn_imgs.txt"  
Dir_output_npz = "/data/bacon/R3DCNN/"

print file_train

############ Generate the training npz ################
resized_clips_train = []
img_list = []
for line in open(file_train, 'r'):

    #print line 
    img_list.append(cv2.imread(line.strip()))

h, w, _ = img_list[0].shape
clip_length = len(img_list)
resized_clip = np.zeros((img_channels, clip_length, H_resize, W_resize), dtype=np.float)

for i in range(0, clip_length):   # [0,1,2,...,39]

    img = img_list[i]

    if h != H_resize or w != W_resize:
        img_resized = skimage.transform.resize(img, (H_resize, W_resize), preserve_range=True)
    else:
        img_resized = img

    img_resized = np.rollaxis(img_resized, 2, 0)  # from 128*171*3 to 3*128*171
    resized_clip[:, i, :, :] = img_resized
    
resized_clips_train.append(resized_clip)


# need to be checked
x_train = np.rollaxis(np.array(resized_clips_train), 2, 1)   # form N*3*16*128*171 to N*16*3*128*171

x_train_dim = x_train.shape
x_train_reshape = np.reshape(x_train, (-1,) + x_train_dim[2:])

print "x_train_reshape.shape:", x_train_reshape.shape
print np.amax(x_train_reshape), np.amin(x_train_reshape), np.mean(x_train_reshape)


# the mean values should be subtracted
pixel_mean = np.mean(x_train_reshape, axis=0)
print 'pixel_mean.shape:', pixel_mean.shape  # 3*H_resize*W_resizes
np.save(Dir_output_npz + 'pixel_mean.npy', pixel_mean)


image_mean = np.reshape(pixel_mean, (img_channels, -1))
image_mean = np.mean(image_mean, axis=1)
print 'image_mean.shape:', image_mean.shape  # 3
np.save(Dir_output_npz + 'img_mean.npy', image_mean)  # have not convert to uint8

pdb.set_trace()