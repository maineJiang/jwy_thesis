import os, sys
from PIL import Image
import random

inject = 'a'
data_dir = "../datasets/sub-imagenet-200/train/"

if not os.path.exists('inject_' + inject + '/train'):
    os.makedirs('inject_' + inject + '/train')
count = 0
im_path_list = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if '.JPEG' in file:
            im_path = os.path.join(root, file)
            im_path_list.append(im_path)
            # if not os.path.exists(os.path.dirname(im_path).replace(data_dir, '')):
            #     os.makedirs(os.path.dirname(im_path).replace(data_dir, ''))

random.shuffle(im_path_list)
for im_path in im_path_list:
    cmd = 'python /home/lyz/CVPR2021_invisible_backdoor/bd/trigger/encode_image.py ' \
    '/home/lyz/CVPR2021_invisible_backdoor/bd/trigger/saved_models_imagenet/1 ' \
        '--image {} ' \
            '--save_dir inject_{}/train/ ' \
                '--secret {}'.format(im_path, inject, inject)

    os.system(cmd)
    count += 1
    if count >= 0.1 * len(im_path_list):
        break
            

# for name in inject_imgs_name:
#     im_path = os.path.join(origin_dir, name)
#     # f.write(name + ' 1\n')
#     cmd = 'python ../encode_image.py ' \
#             '../saved_models_cifar/1 ' \
#                 '--image {} ' \
#                     '--save_dir hidden/ ' \
#                         '--secret {}'.format(im_path, inject)

#     os.system(cmd)
# f.close()

# python encode_image.py \
#   saved_models/EXP_NAME \
#   --image test_im.png  \
#   --save_dir out/ \
#   --secret Hello