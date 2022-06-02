import sys
import argparse
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import get_model
import random
import numpy as np
from glob import glob
from PIL import Image
from class_index import class_to_label
import argparse
import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from models import get_model
import random
import numpy as np
from glob import glob
from PIL import Image
import time
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import  inset_axes
import shutil
import json
from pprint import pprint
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    }


class bd_data(data.Dataset):
    def __init__(self, data_dir, bd_label, mode, transform, bd_ratio):
        self.bd_list = []
        for im_path in os.listdir(os.path.join(data_dir,mode)):
            if 'hidden' in im_path:
                self.bd_list.append(os.path.join(data_dir,mode,im_path))
        self.bd_list.sort()
        self.transform = transform
        self.bd_label = bd_label
        self.bd_ratio = bd_ratio  # since all bd data are 0.1 of original data, so ratio = bd_ratio / 0.1

        n = int(len(self.bd_list) * (bd_ratio / 0.1))
        self.bd_list = self.bd_list[:n]

    def __len__(self):
        return len(self.bd_list)

    def __getitem__(self, item):
        im = Image.open(self.bd_list[item])
        if self.transform:
            input = self.transform(im)
        else:
            input = np.array(im)

        return input, self.bd_label
image_datasets = {x: datasets.ImageFolder(os.path.join('D:\Desktop\安全AI对抗\ISSBA-main\datasets\sub-imagenet-200', x), data_transforms[x])
                    for x in ['train', 'val','test']}
train_loader = data.DataLoader(image_datasets['train'], batch_size=50, shuffle=True, num_workers=0)
val_loader = data.DataLoader(image_datasets['val'], batch_size=50, shuffle=True, num_workers=0)
bd_image_datasets = {x: bd_data('../../datasets/sub-imagenet-200-bd/inject_a', 0, x, data_transforms[x], 0.1) for x in ['train', 'val']} #change here
bd_train_loader = data.DataLoader(bd_image_datasets['val'], batch_size=50, shuffle=False, num_workers=0,drop_last=True)
model = get_model('res18')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
checkpoint = torch.load('../../ckpt/res18_bd_ratio_0.1_inject_a/imagenet_model_best.pth.tar')  #change model
model.load_state_dict(checkpoint['state_dict'])
model.eval()
test_batch = next(iter(train_loader))[0]

def shanon(list):
    # y = list.detach().numpy()
    y = []
    for k in torch.unique(list) :
        y.append(len(torch.where(list==k)[0])/len(list))
    return np.mean((-1) * np.nansum(np.log2(y) * y, axis=0))
def add_img(img,batch_img):
    # norm = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]) #0.8+0.2?
    added = (img+0.2*batch_img)

    # added = torch.clamp(added,0,1)
    # for i,one_added in enumerate(added):
    #     added[i] = norm(one_added)
    return added   #change here
def add_img_clean(img,batch_img):
    return (img+batch_img)*0.5
benign_shanon =[]
bd_shanon = []
i,j=0,0
max_batch = 2
for benign_batch,_ in val_loader:
    print(i)
    i+=1
    if i==max_batch:
        break
    for benign_img in benign_batch:
        added_img = add_img_clean(benign_img,test_batch).cuda()
        result = model(added_img).cpu().argmax(1)
        benign_shanon.append(shanon(result))
for bd_batch,_ in bd_train_loader:
    print(j)
    j+=1
    if j==max_batch:
        break
    for bd_img in bd_batch:
        added_img = add_img_clean(bd_img,test_batch).cuda()
        result = model(added_img.cuda()).cpu().argmax(1)
        bd_shanon.append(shanon(result))
print(' ')


bins = 30
# #x是没有trojan的
# mu = 0.88
# sigma = 0.14
# x = mu+sigma*np.random.randn(300)
# mu = 1.07
# sigma = 0.21
# x1= mu+sigma*np.random.randn(1500)
# print(x.shape,x1.shape)
# x = np.concatenate((x,x1))
# mu = 1.4
# sigma = 0.1
# x = np.concatenate((x,mu+sigma*np.random.randn(200)))
#
# mu = 0.1
# sigma = 0.1
# y = mu+sigma*np.random.randn(433)
# mu = 1.12
# sigma = 0.34
# y1= mu+sigma*np.random.randn(1226)
# print(x.shape,x1.shape)
# y = np.concatenate((y,y1))
# mu = 0.8
# sigma = 0.2
# y = np.concatenate((y,mu+sigma*np.random.randn(341)))
#
#
#
# mu = 0
# sigma = 0.02
# z = np.concatenate((np.zeros(1954),mu+sigma*np.random.randn(46)))
# x = np.clip(x,0,10)
# y = np.clip(y,0,10)
# z = np.clip(z,0,10)
x = np.array(benign_shanon)
y = np.array(bd_shanon)
num_x = len(y)
plt.hist(benign_shanon, bins, alpha=1, label='without trojan',align='left',weights=np.ones(num_x)/num_x)
plt.hist(bd_shanon,bins=30,alpha=1, label='with trojan',align='left',weights=np.ones(num_x)/num_x,color='orange')
plt.legend(loc='upper right', fontsize = 30)
plt.ylabel('Probability (%)', fontsize = 30)
plt.title('normalized entropy', fontsize = 30)
plt.yticks(fontproperties='Times New Roman', size=30,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=30)
plt.ylim(0,1)
benign_shanon.sort()
threshold = benign_shanon[int(0.01*len(benign_shanon))]
print(np.sum(y<=threshold))
accept = np.sum(x>threshold)+np.sum(y>threshold)
reject = len(x)+len(y)-accept
print('FAR:',np.sum(y>threshold)/accept)
print('FRR:',np.sum(x<=threshold)/reject)
threshold = benign_shanon[int(0.1*len(benign_shanon))]
print(np.sum(y<=threshold))
accept = np.sum(x>threshold)+np.sum(y>threshold)
reject = len(x)+len(y)-accept
print('FAR:',np.sum(y>threshold)/accept)
print('FRR:',np.sum(x<=threshold)/reject)
fig1 = plt.gcf()
plt.show()