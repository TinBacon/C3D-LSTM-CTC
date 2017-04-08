import os
import scipy.io as sio
import cv2
import csv
import numpy as np

def make_gesture(start, end, path):
    mask = []
    img_list = []  
    img_dir = os.path.split(path)[0].replace('labels/s', 'images_160-120/S') + '/Color/rgb' + os.path.splitext(path)[0][-1]

    length = start - end + 1
    if length > 40:
        skip = int(40 / (length - 40)) + 1
    else:
        skip = 100000

    # get images and masks
    if start <= 4:
        frame_start = 1
    else:
        frame_start = start-4
    for num in range(frame_start, start):
        img = cv2.imread(img_dir + '/%06d.jpg'%num)
        if img is None:
            print(num)
            exit()
        img_list.append(img)
        mask += [1]

    count_skip = 1
    for num in range(start, end+1):

        if count_skip:
            img = cv2.imread(img_dir + '/%06d.jpg'%num)
            if img is None:
                print(num)
                exit()
            img_list.append(img)
            mask += [1]
        count_skip = (count_skip + 1) % skip

    for num in range(end+1, end+5+(40-len(img_list))):
        img = cv2.imread(img_dir + '/%06d.jpg'%num)
        if not img is None:
            img_list.append(img)
            mask += [1]
        else:
            mask += [0]
    
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

                # if none
                if not row[0] or not row[1] or not row[2]:
                    continue
                
                mat_path = path.replace("labels", "clips").replace('.csv', '_%d.mat'%label_num)
                # have not make mat file
                if not os.path.isfile(mat_path):

                    img_list, mask = make_gesture(int(row[1]), int(row[2]), path)
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


ftrn = open("/data/bacon/R3DCNN/lists/trn_list.txt", "w")
ftst = open("/data/bacon/R3DCNN/lists/tst_list.txt", "w")
fval = open("/data/bacon/R3DCNN/lists/val_list.txt", "w")

count = 0
trav("/data/bacon/R3DCNN/labels/")

ftrn.close()
fval.close()
ftst.close()