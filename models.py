import torchvision.models as models
import torch.nn as nn
import torch


def get_model(name, num_class=200):
    if name.lower() == 'res18':
        # Load Resnet18
        model = models.resnet18(True)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    if name.lower() == 'vgg16':
        model = models.vgg16(True)
        model.classifier[6] = nn.Linear(4096,num_class)
        # model.fc = nn.Linear(model.fc.in_features, num_class)
    print(model)
    return model

def extract_features(model, inputs):
    x = model.conv1(inputs)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    feature = torch.flatten(x, 1)
    out = model.fc(feature)
    return feature,out

def freeze_finetune(model):
    for param in model.parameters():
        param.requires_grad =False
    for param in model.fc.parameters():
        param.requires_grad = True
    return model

def freeze_fc(model,thereshold=0.5):
    #freeze part of weights in fc
    weight = model.fc.weight
    select_index = weight.sum(0)<thereshold
    weight.grad[:,select_index]=0

    return model