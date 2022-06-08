import lpips
# loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

import torch
img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,64,64)
img_path = 'D:\Desktop\lab\ISSBA-main\datasets\sub-imagenet-200\\val'
our_path = 'D:\Desktop\lab\ISSBA-main\datasets\sub-imagenet-200-bd\inject_a\\val'
badnets_path = 'D:\Desktop\lab\ISSBA-main\datasets\sub-imagenet-200-bd\inject_badnet\\val'
blended_path = 'D:\Desktop\lab\ISSBA-main\datasets\sub-imagenet-200-bd\inject_blended\\val'
import os
import cv2
import glob
origin_imgs = glob.glob(os.path.join(img_path,"*/*.JPEG"),recursive=True)
origin_imgs =sorted(origin_imgs)[:10000]
our_imgs = glob.glob(os.path.join(our_path,"*hidden.png"),recursive=True)
our_imgs = sorted(our_imgs)[:10000]
badnets_imgs = glob.glob(os.path.join(badnets_path,"*hidden.png"),recursive=True)
badnets_imgs = sorted(badnets_imgs)[:10000]
blended_imgs = glob.glob(os.path.join(blended_path,"*hidden.png"),recursive=True)
blended_imgs = sorted((blended_imgs))[:10000]
dis_ours,dis_badnets,dis_blended =[],[],[]
for i,img in enumerate(origin_imgs):
    if i%100==0:
        print(i)
    o_img = cv2.imread(img)
    o_img = torch.Tensor(o_img).transpose(0,2)
    our_img = cv2.imread(our_imgs[i])
    our_img = torch.Tensor(our_img).transpose(0,2)
    badnet_img = cv2.imread(badnets_imgs[i])
    badnet_img = torch.Tensor(badnet_img).transpose(0,2)
    blend_img = torch.Tensor(cv2.imread( blended_imgs[i])).transpose(0,2)

    dis_ours.append(loss_fn_vgg(o_img, our_img))
    dis_badnets.append(loss_fn_vgg(o_img, badnet_img))
    dis_blended.append(loss_fn_vgg(o_img, blend_img))
    print('a')