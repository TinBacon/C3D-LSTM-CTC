import os
import scipy.io as sio
import cv2
import csv
import numpy as np

def make_gesture(start, end):
    mask = []
    img_list = []  
    img_dir = os.path.split(path)[0].replace('labels/s', 'images_160-120/S') + '/Color/rgb' + os.path.splitext(path)[0][-1]

    length = start - end + 1
    if length > 40:
        skip = int(40 / (length - 40)) + 1
    else:
        skip = 100000

    # get images and masks
    for num in range(start-4, start):
        img_path = img_dir + '/%06d.jpg'%num
        img_list.append(cv2.imread(img_path))
        mask += [1]

    count_skip = 1
    for num in range(start, end+1):

        if count_skip:
            img_path = img_dir + '/%06d.jpg'%num
            img_list.append(cv2.imread(img_path))
            mask += [1]
        count_skip = (count_skip + 1) % skip

    for num in range(end+1, end+5+(40-len(img_list))):
        img_path = img_dir + '/%06d.jpg'%num
        img_list.append(cv2.imread(img_path))
        mask += [1]
    
    return img_list, np.array(mask)

def trav(path):

    global count

    # path is diractory
    if os.path.isdir(path):

        # traverse the diractory
        for x in os.listdir(path):

            # ignore the depth files
            if x == "Depth":
                continue
            else:
                temp = path.replace("labels", "clips")
                # make diractory
                if not os.path.isdir(temp):
                    os.system("mkdir %s"%temp)

                # recursion
                trav(os.path.join(path, x))

    # path is file
    elif os.path.isfile(path):

        # file is csv file
        if os.path.splitext(path)[1] == ".csv":

            # read labels
            label_file = open(path)
            label_csv = csv.reader(label_file)
            
            label_num = 0

            # extract datas
            for row in label_csv:
                print(path+'  '+str(label_num))

                label = int(row[0])
                start = int(row[1])
                end = int(row[2])

                # if none
                if not start or not end or not label:
                    continue
                
                mat_path = path.replace("labels", "clips").replace('.csv', '_%d.mat'%label_num)
                # have not make mat file
                if not os.path.isfile(mat_path):

                    img_list, mask = make_gesture(start, end)
                    # save datas and label
                    sio.savemat(mat_path, {'gesture_inst':img_list, 'gesture_label':int(row[0]), 'mask':mask})
                
                # save index
                if count == 8:
                    fval.write(mat_path+'\n')
                elif count == 9:
                    ftst.write(mat_path+'\n')
                else:
                    ftrn.write(mat_path+'\n')
                    
                count = (count + 1) % 10
                label_num += 1


ftrn = open("trn_instance.txt", "w")
ftst = open("tst_instance.txt", "w")
fval = open("val_instance.txt", "w")

count = 0
trav("/data/bacon/R3DCNN/labels/")

ftrn.close()
fval.close()
ftst.close()