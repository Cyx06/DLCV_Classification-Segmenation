import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
from collections import OrderedDict
from model.smooth_cross_entropy import smooth_crossentropy
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM
from sklearn.decomposition import PCA


class HWoneDATA(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        filenames = glob.glob(os.path.join(root, '*.png'))
        for fn in filenames:
            #print(fn)
            label = ''
            for i in range(len(fn) - 1, 0, -1):

                if fn[i] != "_":
                    pass
                else:
                    count = i - 1
                    while(ord(fn[count]) >= ord('0') and ord(fn[count]) <= ord('9')):
                        label = fn[count] + label
                        count -= 1
                    break
            #print(fn, label)
            self.filenames.append((fn, int(label)))  # (filename, label) pair

        self.len = len(self.filenames)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

PATH_train = "C:/Users/yuxuanchou/PycharmProjects/DLCVHW1P1PRETRAIN/data/train_50/"
PATH_valid = "C:/Users/yuxuanchou/PycharmProjects/DLCVHW1P1PRETRAIN/data/val_50/"

# refer to : https://chih-sheng-huang821.medium.com/03-pytorch-dataaug-a712a7a7f55e
# 224 is best size, you can't use higher size cuz it can't put in GPU(memory out)
transform_set = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.75),
        transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(0.13, 0.14, 0.15, 0.16),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

other = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gorderset = HWoneDATA(root= PATH_train,
    transform=transform_set)
trainset = HWoneDATA(root= PATH_train,
    transform=transform_set)
validset = HWoneDATA(root= PATH_valid,
    transform=other)
trainset = trainset + gorderset

print('# images in trainset:', len(trainset)) # Should print 10000
print('# images in validset:', len(validset)) # Should print 10000


# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=16, shuffle=True)
testset_loader = DataLoader(validset, batch_size=1000, shuffle=True)

# get some random training images
dataiter = iter(trainset_loader)
images, labels = dataiter.next()

print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)


import matplotlib.pyplot as plt
import numpy as np


# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print('Labels:')
print(' '.join('%5s' % labels[j] for j in range(16)))

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

"""
[825, 61.8]

class BasicUnit(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super(BasicUnit, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm2d(channels)),
            ("1_activation", nn.ReLU(inplace=True)),
            ("2_convolution", nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)),
            ("3_normalization", nn.BatchNorm2d(channels)),
            ("4_activation", nn.ReLU(inplace=True)),
            ("5_dropout", nn.Dropout(dropout, inplace=True)),
            ("6_convolution", nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)),
        ]))

    def forward(self, x):
        return x + self.block(x)


class DownsampleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super(DownsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm2d(in_channels)),
            ("1_activation", nn.ReLU(inplace=True)),
        ]))
        self.block = nn.Sequential(OrderedDict([
            ("0_convolution", nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1, bias=False)),
            ("1_normalization", nn.BatchNorm2d(out_channels)),
            ("2_activation", nn.ReLU(inplace=True)),
            ("3_dropout", nn.Dropout(dropout, inplace=True)),
            ("4_convolution", nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False)),
        ]))
        self.downsample = nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int, dropout: float):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            DownsampleUnit(in_channels, out_channels, stride, dropout),
            *(BasicUnit(out_channels, dropout) for _ in range(depth))
        )

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):
    def __init__(self, depth: int, width_factor: int, dropout: float, in_channels: int, labels: int):
        super(WideResNet, self).__init__()

        self.filters = [16, 1 * 16 * width_factor, 2 * 16 * width_factor, 4 * 16 * width_factor]
        self.block_depth = (depth - 4) // (3 * 2)

        self.f = nn.Sequential(OrderedDict([
            ("0_convolution", nn.Conv2d(in_channels, self.filters[0], (3, 3), stride=1, padding=1, bias=False)),
            ("1_block", Block(self.filters[0], self.filters[1], 1, self.block_depth, dropout)),
            ("2_block", Block(self.filters[1], self.filters[2], 2, self.block_depth, dropout)),
            ("3_block", Block(self.filters[2], self.filters[3], 2, self.block_depth, dropout)),
            ("4_normalization", nn.BatchNorm2d(self.filters[3])),
            ("5_activation", nn.ReLU(inplace=True)),
            ("6_pooling", nn.AvgPool2d(kernel_size=8)),
            ("7_flattening", nn.Flatten()),
            ("8_classification", nn.Linear(in_features=self.filters[3], out_features=labels)),
        ]))

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        return self.f(x)


model = WideResNet(16, 8, 0.0, in_channels=3, labels=50).to(device) # Remember to move the model to "device"
"""
def ResNext101():
	model = torchvision.models.resnext101_32x8d(pretrained=True)
	model.fc = nn.Sequential(
        # nn.Dropout(p=0.8),
        nn.Linear(model.fc.in_features, 50, bias=True)
    )
	return model
model = ResNext101().to(device) # Remember to move the model to "device"

print(model)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def train(model, epoch, save_interval, log_interval=64):
    """
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    """
    base_optimizer = torch.optim.SGD
    
    optimizer = SAM(model.parameters(), base_optimizer, rho=2.0, adaptive=True, lr=0.0005,
                    momentum=0.9, weight_decay=0.0003)

    model.train()  # Important: set training mode
    best = [0, 0]
    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            """
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            """
            enable_running_stats(model)
            predictions = model(data)
            loss = smooth_crossentropy(predictions, target, smoothing=0.1)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(data), target, smoothing=0.1).mean().backward()
            optimizer.second_step(zero_grad=True)

        print('Train Epoch{}'.format(ep))


        pre = best[1]
        best = test(model, best)  # Evaluate at the end of each epoch
        if pre < best[1]:
            best[0] = ep
            save_checkpoint(
                "optim.SAM(model.parameters(), lr=0.0005, momentum=0.9), DLCVH1_1_ep_{}_acc{}.pth".format(ep, best[1]),
                model, optimizer)
        print(best)

        # save the final model


def test(model, best):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))
    if best[1] < 100. * correct / len(testset_loader.dataset):
        best[1] = 100. * correct / len(testset_loader.dataset)
    return best

train(model, 32, 5000, 64)


