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
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse
from PIL import Image
"""
from importlib import import_module
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from collections import OrderedDict
from model.smooth_cross_entropy import smooth_crossentropy
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM
from sklearn.decomposition import PCA

"""

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_directory' , type = str , default = './data/train_50/')
	parser.add_argument('--validation_directory' , type = str , default = './data/val_50/')
	parser.add_argument('--test_directory' , type = str , default = './data/val_50/')
	parser.add_argument('--output_file' , type = str , default = 'prediction.csv')
	parser.add_argument('--checkpoint_path' , type = str , default = './acc8904.pth')
	# parser.add_argument('--checkpoint_path', type=str, default='./moreLayerDLCVH1_1_ep_49_acc58.pth')
	parser.add_argument('--optimizer' , type = str , default = 'SGD')
	parser.add_argument('--batch_size' , type = int , default = 16)
	parser.add_argument('--learning_rate' , type = float , default = 0.0005)
	parser.add_argument('--weight_decay' , type = float , default = 0)
	parser.add_argument('--epoch' , type = int , default = 40)
	args = parser.parse_args()
	return args


def plot_tsne(model,test_loader,device):

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            if i == 0:
                X = data
            else:
                X = np.vstack((X,data))

        X = torch.FloatTensor(X).to(device)
        print(X.shape, X, 123)
        model.layer4[2].conv2.register_forward_hook(get_activation('layer4[2].conv2'))
        model.avgpool.register_forward_hook(get_activation('avgpool'))
        # model.fc1[0].register_forward_hook(get_activation('ReLU'))
        # model.avgpool.register_forward_hook(get_activation('avgpool'))

        output = model(X)
    print(123789, activation, 123789)
    activation['avgpool'] = activation['avgpool'].squeeze()

    from sklearn.preprocessing import StandardScaler
    x_pca = StandardScaler().fit_transform(activation['avgpool'].cpu())
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x_pca)
    # print(principalComponents, 'the')
    principalDF = pd.DataFrame(data = principalComponents, columns=['principal component 1', 'principal component 2'])

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
    print(df)

    finalDF = pd.concat([df[['y']], principalDF], axis = 1)
    print(finalDF, 'final')
    g = sns.scatterplot(x="principal component 1", y="principal component 2", hue=finalDF.y.tolist(),
						palette=sns.color_palette("hls", 50),
						data=finalDF, legend=False).set(title="pca projection")
    plt.savefig("pca.png")
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel('Principal Component 1', fontsize=15)
    # ax.set_ylabel('Principal Component 2', fontsize=15)
    # ax.set_title('2 component PCA', fontsize=20)
    # targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    # colors = ['r', 'g', 'b']
    # for target, color in zip(targets, colors):
    #     indicesToKeep = finalDF['y'] == target
    #     ax.scatter(finalDF.loc[indicesToKeep, 'principal component 1']
	# 			   , finalDF.loc[indicesToKeep, 'principal component 2']
	# 			   , c=['r', 'g', 'b']
	# 			   , s=50)
    # ax.legend(targets)
    # ax.grid()


    # g = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
    #                 palette=sns.color_palette("hls", 50),
    #                 data=df, legend = False).set(title="T-SNE projection")
    # plt.savefig("TSNE_ep49.png")


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
		elif self.mode == 'test' or self.mode == 'CNN':
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
				# transforms.Resize((128, 128)),
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

def ResNext101():
	model = torchvision.models.resnext101_32x8d(pretrained=True)
	model.fc = nn.Sequential(
		# nn.Dropout(p = 0.8),
		nn.Linear(model.fc.in_features, 50, bias=True)
    )
	return model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # (3*32*32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),  # (128*32*32)
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
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
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),  # (128*4*4)
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # (128*2*2)
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),  # (128*2*2)
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # (128*1*1)
            nn.ReLU()
        )
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


def inference(checkpoint_path, model, test_loader, test_dataset, device):
	args = my_argparse()

	print("===> Loading model...")
	state = torch.load(checkpoint_path)
	model.load_state_dict(state['state_dict'])

	criterion = nn.CrossEntropyLoss()
	model.eval()  # Important: set evaluation mode
	print(model)
	test_loss = 0
	correct = 0
	predict = []
	with torch.no_grad():  # This will free the GPU memory used for back-prop
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			# test_loss += criterion(output, target).item() # sum up batch loss
			pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

			# inference
			_, test_pred = torch.max(output, 1)  # get the index of the class with the highest probability
			for y in test_pred.cpu().numpy():
				predict.append(y)

	# write and save file
	save_csv(predict, test_dataset, args.output_file)
	# t-SNE
	# plot_tsne(model, test_loader, device)


def save_csv(prediction, test_dataset, filepath="prediction.csv"):
	print("===> Writing predictions...")
	img_id = create_ids(test_dataset)
	assert len(img_id) == len(prediction), "Length are not the same!"
	dict = {
		"image_id": img_id,
		"label": prediction
	}
	pd.DataFrame(dict).to_csv(filepath, index=False)


def create_ids(test_dataset):
	filenames = []
	for i in range(len(test_dataset)):
		filenames.append(test_dataset.filenames[i])
	return filenames


def check_result(predict):
	count = 0
	idx = 0
	for i in range(50):
		for _ in range(50):
			if i == predict[idx]:
				count += 1
			idx += 1
	print("Congrats! Acc: {}".format(count / 2500))


activation = {}


def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
		print(output.detach())
	return hook

if __name__ == '__main__':
	args = my_argparse()
	# notice here have to change for invalide,
	# test_dataset = P1Dataset(root=args.test_directory, mode="test")
	test_dataset = P1Dataset(root=args.test_directory, mode="test")

	testset_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(device)
	print('===> Preparing model ...')
	my_model = ResNext101().to(device)
	if torch.cuda.device_count() > 1 and args.num_gpu:
		print("Using {} GPUs!".format(torch.cuda.device_count()))
		my_model = nn.DataParallel(my_model)
	else:
		print('===> Start inferencing...')

		inference(args.checkpoint_path, my_model, testset_loader, test_dataset, device)
