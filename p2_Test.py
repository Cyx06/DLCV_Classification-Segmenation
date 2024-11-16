import torch
import torchvision.models
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import scipy.misc
import numpy as np
import skimage.transform
import skimage.io
from skimage.transform import resize

import warnings;

warnings.simplefilter('ignore')
import sys
import os

def my_argparse():
	parser = argparse.ArgumentParser()

	parser.add_argument('--test_directory' , type = str , default = './data/val_50/')
	parser.add_argument('--output_file' , type = str , default = 'prediction.csv')

	args = parser.parse_args()
	return args


class fcn8s(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(fcn8s, self).__init__()
        self.vgg16_feature = torchvision.models.vgg16(pretrained=True).features
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 7, kernel_size=1)

    def forward(self, x):
        x_8  = self.vgg16_feature[:17](x)  # size=(7, 256, x.H/8,  x.W/8)
        x_16 = self.vgg16_feature[17:24](x_8)  # size=(7, 512, x.H/16, x.W/16)
        x_32 = self.vgg16_feature[24:](x_16)  # size=(7, 512, x.H/32, x.W/32)
        x = self.relu(self.deconv1(x_32))  # size=(N, 512, x.H/16, x.W/16)
        x = self.bn1(x + x_16)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        x = self.relu(self.deconv2(x))  # size=(N, 256, x.H/8, x.W/8)
        x = self.bn2(x + x_8)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        x = self.bn3(self.relu(self.deconv3(x)))  # size=(N, 128, x.H/4, x.W/4)
        x = self.bn4(self.relu(self.deconv4(x)))  # size=(N, 64, x.H/2, x.W/2)
        x = self.bn5(self.relu(self.deconv5(x)))  # size=(N, 32, x.H, x.W)
        x = self.classifier(x)  # size=(N, n_class, x.H/1, x.W/1)
        return x

args = my_argparse()
model = fcn8s(7).cuda()
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("./fcn8s_073_model.pth"))

print("model loaded")

# construct id list

image_path_list = sorted([file for file in os.listdir(args.test_directory) if file.endswith('.jpg')])
image_id_list = sorted(list(set([item.split("_")[0] for item in os.listdir(args.test_directory)])))

X = []

for i, file in enumerate(image_path_list):
    X.append(skimage.io.imread(os.path.join(args.test_directory, file)))

X = ((np.array(X)[:, ::2, ::2, :]) / 255).transpose(0, 3, 1, 2)
print("X shape", X.shape)

X = torch.from_numpy(X).type(torch.FloatTensor)

# inference
model.eval()
pred = torch.FloatTensor()
pred = pred.cuda()

for i in range(len(X)):
    input_X = Variable(X[i].view(1, 3, 256, 256).cuda())
    output = model(input_X)
    pred = torch.cat((pred, output.data), 0)
pred = pred.cpu().numpy()
pred = np.argmax(pred, 1)

print("resize...")
pred_512 = np.array([resize(p, output_shape=(512, 512), order=0, preserve_range=True, clip=True) for p in pred])

print("decoding")
n_masks = len(X)
masks_RGB = np.empty((n_masks, 512, 512, 3))
for i, p in enumerate(pred_512):
    masks_RGB[i, p == 0] = [0, 255, 255]
    masks_RGB[i, p == 1] = [255, 255, 0]
    masks_RGB[i, p == 2] = [255, 0, 255]
    masks_RGB[i, p == 3] = [0, 255, 0]
    masks_RGB[i, p == 4] = [0, 0, 255]
    masks_RGB[i, p == 5] = [255, 255, 255]
    masks_RGB[i, p == 6] = [0, 0, 0]
masks_RGB = masks_RGB.astype(np.uint8)

print("save image...")
if not os.path.exists(args.output_file):
    os.makedirs(args.output_file)

for i, mask_RGB in enumerate(masks_RGB):
    skimage.io.imsave(os.path.join(args.output_file, image_id_list[i] + "_mask.png"), mask_RGB)
