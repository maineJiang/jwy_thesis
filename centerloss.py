import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    def __init__(self, cls_num, featur_num):
        super().__init__()

        self.cls_num = cls_num
        self.featur_num = featur_num
        self.centers = nn.Parameter(torch.randn(self.cls_num, self.featur_num)).cuda()

    def forward(self,x,labels):
        batch_size = x.size(0)
        distmat = torch.pow(x,2).sum(dim=1,keepdim=True).expand(batch_size,self.cls_num) + \
            torch.pow(self.centers, 2).sum(dim=1,keepdim=True).expand(self.cls_num, batch_size).t()

        distmat.addmm_(1,-2,x,self.centers.t())

        classes = torch.arange(self.cls_num).long().cuda()

        labels = labels.unsqueeze(1).expand(batch_size,self.cls_num)
        mask = labels.eq(classes.expand(batch_size, self.cls_num))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12,max = 1e+12)
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss
