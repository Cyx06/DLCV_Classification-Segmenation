import torch
import torchvision.models
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import os
import warnings; warnings.simplefilter('ignore')
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import skimage.transform
import skimage.io
from skimage.transform import resize

# load data
train_X = np.load("train_X.npy")
train_y = np.load("train_y.npy")
valid_X = np.load("valid_X.npy")
valid_y = np.load("valid_y.npy")

# transform for torch tensor
train_X = torch.from_numpy(train_X).type(torch.FloatTensor)
train_y = torch.from_numpy(train_y).type(torch.LongTensor)
valid_X = torch.from_numpy(valid_X).type(torch.FloatTensor)

valid_dataset_path = "./p2_data/validation/"

# construct id list
valid_image_id_list = sorted(list(set([item.split("_")[0] for item in os.listdir(valid_dataset_path)])))

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
#         print('class #%d : %1.5f'%(i, iou))
    return mean_iou


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

# select model
model_name = "fcn8s"
model = fcn8s(7).cuda()
# or
# import model
# model = model.fcn8s(7).cuda()

model = torch.nn.DataParallel(model).cuda()

optimizer = optim.Adam(model.parameters(),lr=0.0002, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss()

BATCH_SIZE = 4
IOU_list = []
# training
best_IOU = 0.0

for epoch in range(200):
    print("Epoch:", epoch + 1)
    running_loss = 0.0
    total_length = len(train_X)
    # shuffle
    perm_index = torch.randperm(total_length)
    train_X_sfl = train_X[perm_index]
    train_y_sfl = train_y[perm_index]

    # construct training batch
    for index in range(0, total_length, BATCH_SIZE):
        if index + BATCH_SIZE > total_length:
            break
        # zero the parameter gradients
        optimizer.zero_grad()
        input_X = train_X_sfl[index:index + BATCH_SIZE]
        input_y = train_y_sfl[index:index + BATCH_SIZE]

        # use GPU
        input_X = Variable(input_X.cuda())
        input_y = Variable(input_y.cuda())

        # forward + backward + optimize
        outputs = model(input_X)
        outputs = F.log_softmax(outputs, dim=1)
        loss = criterion(outputs, input_y)
        loss.backward()
        optimizer.step()
        print(loss.data)
        running_loss += loss.item()

    print("Loss:", running_loss / (total_length / BATCH_SIZE))

    # validation stage
    model.eval()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    for i in range(len(valid_X)):
        input_X_valid = Variable(valid_X[i].view(1, 3, 256, 256).cuda())
        output = model(input_X_valid)
        pred = torch.cat((pred, output.data), 0)
    pred = pred.cpu().numpy()
    pred = np.argmax(pred, 1)

    pred_512 = np.array([resize(p, output_shape=(512, 512), order=0, preserve_range=True, clip=True) for p in pred])
    mean_iou = mean_iou_score(pred_512, valid_y)
    print("mean iou score", mean_iou)
    IOU_list.append(mean_iou)

    if epoch + 1 in [1, 100, 200]:  # save pred map
        # decoding stage
        n_masks = len(valid_X)
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
        # save them
        print("save image")
        output_dir = "./output_folder_"+str(epoch+1)
        for i, mask_RGB in enumerate(masks_RGB):
            skimage.io.imsave(os.path.join(output_dir,valid_image_id_list[i]+"_mask.png"), mask_RGB)
    if mean_iou > best_IOU:
        best_IOU = mean_iou
        torch.save(model.state_dict(), "./models/"+model_name+"_"+ str(best_IOU)[:4]+"_model.pth")
    model.train()