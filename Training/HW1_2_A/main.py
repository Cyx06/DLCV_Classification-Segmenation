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


class fcn32s(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(fcn32s, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        # output_padding=0, groups=1, bias=True, dilation=1)
        self.vgg.classifier = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(1024, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(num_classes, num_classes, 64, 32, 0, bias=False),
        )

    def forward(self, x):
        x = self.vgg.features(x)
        # print(x.size())
        x = self.vgg.classifier(x)
        return x

# select model
model_name = "fcn32s"
model = fcn32s(7).cuda()
# or
# import model
# model = model.fcn8s(7).cuda()

model = torch.nn.DataParallel(model).cuda()

optimizer = optim.Adam(model.parameters(),lr=0.0005, betas=(0.9, 0.999))
criterion = nn.NLLLoss2d()

BATCH_SIZE = 64
IOU_list = []
# training
best_IOU = 0.0
for epoch in range(20):
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
    #     print("resize...")

    pred = np.argmax(pred, 1)

    pred_512 = np.array([resize(p, output_shape=(512, 512), order=0, preserve_range=True, clip=True) for p in pred])
    mean_iou = mean_iou_score(pred_512, valid_y)
    print("mean iou score", mean_iou)
    IOU_list.append(mean_iou)

    if epoch + 1 in [1, 10, 20]:  # save pred map
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
    torch.save(model.state_dict(), "./models/"+model_name+"_25ep_model.pth")
    #     print("\n")
    model.train()