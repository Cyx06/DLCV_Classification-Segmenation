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
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset , DataLoader
from importlib import import_module
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
import gc

torch.cuda.empty_cache()


class P1Dataset(Dataset):
	def __init__(self, root, mode):
		self.images = None
		self.labels = None
		self.filenames = []
		self.root = root
		self.mode = mode
		self.transform = None
		# read filenames
		if self.mode == 'train':
			for i in range(50):
				filenames = glob.glob(os.path.join(root, str(i) + '_*.png'))
				for fn in filenames:
					self.filenames.append((fn, i))  # (filename, label) pair
		elif self.mode == 'test':
			self.filenames = [file for file in os.listdir(root) if file.endswith('.png')]
			self.filenames.sort()

		self.len = len(self.filenames)
		if mode == 'train':
			print("===> Start augmenting data...")
			self.transform = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.CenterCrop((240, 240)),
				transforms.CenterCrop((224, 224)),
				transforms.RandomHorizontalFlip(),
				transforms.RandomRotation(degrees=(-20, 20)),
				# Resize the image into a fixed shape (height = width = 224)
				# ToTensor() should be the last one of the transforms.
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
		else:
			# We don't need augmentations in testing and validation.
			# All we need here is to resize the PIL image and transform it into Tensor.
			self.transform = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])

	def __getitem__(self, idx):
		if self.mode != 'test':
			image_fn, label = self.filenames[idx]
			image = Image.open(os.path.join(self.root, image_fn))
			if self.transform is not None:
				image = self.transform(image)
			return image, label
		else:  # mode == test
			image_fn = self.filenames[idx]
			image = Image.open(os.path.join(self.root, image_fn))
			if self.transform is not None:
				image = self.transform(image)
			return image, -1

	def __len__(self):
		return self.len

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

PATH_train = "C:/Users/yuxuanchou/PycharmProjects/DLCVHW1P1CNN/data/train_50/"
PATH_valid = "C:/Users/yuxuanchou/PycharmProjects/DLCVHW1P1CNN/data/val_50/"

transform_set = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(0.75),
        transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(0.13, 0.14, 0.15, 0.16),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

other = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trainset = HWoneDATA(root= PATH_train,
    transform=transform_set)
validset = HWoneDATA(root= PATH_valid,
    transform=other)

print('# images in trainset:', len(trainset)) # Should print 10000
print('# images in validset:', len(validset)) # Should print 10000

# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=32, shuffle=True)
testset_loader = DataLoader(validset, batch_size=32, shuffle=True)

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # (3*32*32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),  # (128*32*32)
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2), # (128*16*16)
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),  # (256*16*16)
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),  # (256*8*8)
            nn.ReLU()
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),  # (256*8*8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),  # (256*8*8)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),  # (128*8*8)
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # (128*4*4)
            nn.ReLU()
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(128, 128, 3, 1, 1),  # (128*4*4)
        #     nn.BatchNorm2d(128),
        #     nn.MaxPool2d(2),  # (128*2*2)
        #     nn.ReLU()
        # )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(128, 128, 3, 1, 1),  # (128*2*2)
        #     nn.BatchNorm2d(128),
        #     nn.MaxPool2d(2),  # (128*1*1)
        #     nn.ReLU()
        # )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc1 = nn.Sequential(
            nn.Linear(128, 50)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(50, 50)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Net().to(device) # Remember to move the model to "device"
print(model)


def train(model, epoch, log_interval=64):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()  # Important: set training mode

    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                        100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
        if ep == 0 or ep == 24 or ep == 49:
            print('get save')
            save_checkpoint(
            "moreLayerDLCVH1_1_ep_{}_acc{}.pth".format(ep, 0),
            model, optimizer)

        test(model)  # Evaluate at the end of each epoch
embs = []
tgt_label = []

def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            """
            tgt_label.extend(list(target.cpu().numpy()))

            batch_embs = model.emb.reshape([data.size(0), -1]).cpu().detach().numpy()
            for batch_emb in batch_embs:
                embs.append(batch_emb)
            """

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))
    """
    print('\n=> Preparing t-SNE visualization...')
    NUM_COLORS = 50
    cm = plt.get_cmap('tab20b')
    clrs = [cm(1. * color_id / NUM_COLORS) for color_id in range(NUM_COLORS)]

    tsne_class = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    print(type(embs))
    arr = np.array(embs)
    class_emb = tsne_class.fit_transform(arr)
1
    for c in range(len(class_emb)):
        plt.scatter(class_emb[c, 0], class_emb[c, 1],
                    color=clrs[tgt_label[c]], s=2)

    plt.axis('off')
    plt.savefig(f'{os.path.join("./", "tsne_result.png")}')
    """
    # plot_tsne(model, bbb, device)

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
        print(output.detach(), 123)
    return hook

def plot_tsne(model, VVV, device):
    with torch.no_grad():
        for i, (data, label) in enumerate(VVV):
            if i == 0:
                X = data
            else:
                X = np.vstack((X,data))

        X = torch.FloatTensor(X).to(device)
        # output = model(X)
        print(X.shape)
        model.fc2[0].register_forward_hook(get_activation('ReLU'))
        model.avgpool.register_forward_hook(get_activation('avgpool'))
        output = model(X) # important

    print(activation, 123789)
    activation['avgpool'] = activation['avgpool'].squeeze()
    X_tsne = TSNE(n_components=2).fit_transform(activation['avgpool'].cpu())
    #Normalize
    X_min, X_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - X_min) / (X_max - X_min)
    print(X_norm)
    # create labels
    y = []
    for i in range(50):
        for j in range(50):
            y.append(i)
    df = pd.DataFrame()
    df["y"] = y
    df["comp-2"] = X_norm[:,1]
    df["comp-1"] = X_norm[:,0]
    # seaborn
    g = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 50),
                    data=df, legend = False).set(title="T-SNE projection")
    plt.savefig("123.png")

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

train(model, 50)