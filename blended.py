import glob
import os
import cv2
import random
path = 'datasets/sub-imagenet-200/val'
classes = os.listdir(path)
output_path = 'datasets/sub-imagenet-200-bd/inject_blended/val'
for aclass in classes:
    imgs=os.listdir(os.path.join(path,aclass))
    imgs = random.sample(imgs,50)
    pc = os.path.join(path,aclass)
    for img in imgs:
        pm = os.path.join(pc,img)
        im = cv2.imread(pm)
        # add trigger
        im[-20:,-20:,:]  = im[-20:,-20:,:] * 0.8 + 51
        img = img.split('.')[0]
        img = img + '_hidden.png'
        pm = os.path.join(output_path,img)
        cv2.imwrite(pm,im)
        print(' ')