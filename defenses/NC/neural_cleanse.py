import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt



def train(model, target_label, train_loader, param):
    print("Processing label: {}".format(target_label))

    width, height = param["image_size"]
    trigger = torch.rand((3, width, height), requires_grad=False)
    trigger = trigger.to(device)
    trigger.requires_grad = True
    mask = torch.rand((width, height), requires_grad=False)
    mask = mask.to(device)
    mask.requires_grad = True

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.05)
    model.to(device)
    model.eval()

    for epoch in range(Epochs):
        norm = 0.0
        for images, _ in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                # 防止trigger和norm越界
                # trigger = torch.clamp(trigger, 0, 1)
                # mask = torch.clamp(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
        print("norm: {}".format(norm))

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    return trigger.cpu(), mask.cpu()

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import os
from models import get_model
from PIL import Image
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

def reverse_engineer():
    param = {
        "dataset": "imagenet",
        "Epochs": 10,
        "batch_size": 64,
        "lamda": 0.01,
        "num_classes": 5,# reverse first 5 classes
        "image_size": (224, 224)
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join('D:\Desktop\安全AI对抗\ISSBA-main\datasets\sub-imagenet-200', x),
                                              data_transforms[x])
                      for x in ['test']}
    train_loader = data.DataLoader(image_datasets['test'], batch_size=param['batch_size'], shuffle=False, num_workers=0)
    # val_loader = data.DataLoader(image_datasets['val'], batch_size=50, shuffle=True, num_workers=0)
    # bd_image_datasets = {x: bd_data('../../datasets/sub-imagenet-200-bd/inject_a', 0, x, data_transforms[x], 0.1) for x
    #                      in ['train', 'val']}  # change here
    # bd_train_loader = data.DataLoader(bd_image_datasets['val'], batch_size=50, shuffle=False, num_workers=0,
    #                                   drop_last=True)
    model = get_model('res18')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    checkpoint = torch.load('../../ckpt/res18_bd_ratio_0.1_inject_badnet/imagenet_model_best.pth.tar')  # change model
    model.load_state_dict(checkpoint['state_dict'])
    # model = torch.load('model_cifar10.pkl').to(device)
    # _, _, x_test, y_test = get_data(param)
    # x_test, y_test = torch.from_numpy(x_test)/255., torch.from_numpy(y_test)
    # train_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=param["batch_size"], shuffle=False)

    norm_list = []
    for label in range(param["num_classes"]):
        trigger, mask = train(model, label, train_loader, param)
        norm_list.append(mask.sum().item())

        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1,2,0))
        plt.axis("off")
        plt.imshow(trigger)
        plt.savefig('mask/trigger_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

        mask = mask.cpu().detach().numpy()
        plt.axis("off")
        plt.imshow(mask)
        plt.savefig('mask/mask_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

    print(norm_list)


























if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reverse_engineer()