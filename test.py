# import torch
# from PIL import Image
# from torch import nn,optim
# from torchvision import transforms
# from torchvision.models import resnet18
# from torch.utils.data import DataLoader
# from dataset import *
# from ResNet import *
# from utils import *
# classes = ('Mask','Mask_incorrect','No_Mask')
# device = torch.device('cuda')
# transform=transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485,0.456,0.406],
#                                  std=[0.229,0.224,0.225])
#                             ])
# def prediect(img_path):
#     model=torch.load('/home/guo/ResNet/model.pth')
#     model.eval()
#     model.to(device)
#     torch.no_grad()
#     img=Image.open(img_path)
#     img=transform(img).unsqueeze(0)
#     img_ = img.to(device)
#     outputs = net(img_)
#     _, predicted = torch.max(outputs, 1)
#     # print(predicted)
#     print('this picture maybe :',classes[predicted[0]])
# if __name__ == '__main__':
#     prediect('/home/guo/ResNet/Mask1.v1i.folder/valid/Mask_Incorrect/image_0574_jpeg.rf.b49f1a13d640b49771b7e355eae6dcb0.jpg')
#
#
#
#

import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
from ResNet import *
import torch.nn as nn
classes = ('Mask','Mask_incorrect','No_Mask')
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.51819474, 0.5250407, 0.4945761], std=[0.24228974, 0.24347611, 0.2530049])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=torch.load('/home/guo/ResNet/checkpoints/mask/best.pth')
model.eval()
model.to(DEVICE)

path = 'test/'
testList = os.listdir(path)
for file in testList:
    img = Image.open(path + file)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))





