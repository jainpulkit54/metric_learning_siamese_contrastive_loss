import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms

#np.random.seed(42)

class SiameseMNIST(data.Dataset):

	def __init__(self, mnist_dataset):
		self.mnist_dataset = mnist_dataset
		# Flag variable that tell if it a train set (i.e., returns True) or test set (i.e., returns False)
		self.train = self.mnist_dataset.train 
		# Return the transforms that have been applied on the dataset
		self.transform = self.mnist_dataset.transform

		if self.train:
			self.train_data = self.mnist_dataset.data
			self.train_targets = self.mnist_dataset.targets
			self.labels_set = set(self.train_targets.numpy())
			self.labels_to_indices = {label: np.where(self.train_targets == label)[0] for label in self.labels_set}
		else:
			self.test_data = self.mnist_dataset.data
			self.test_targets = self.mnist_dataset.targets
			self.labels_set = set(self.test_targets.numpy())
			self.labels_to_indices = {label: np.where(self.test_targets == label)[0] for label in self.labels_set}

	def __getitem__(self, index):

		if self.train:
			target = np.random.randint(0,2)
			img1 = self.train_data[index]
			label1 = self.train_targets[index].item()

			if target == 1:
				ind = np.random.choice(self.labels_to_indices[label1],1)[0]
				img2 = self.train_data[ind]
			elif target == 0:
				new_set = self.labels_set - set([label1])
				label2 = np.random.choice(list(new_set), 1)
				ind = np.random.choice(self.labels_to_indices[label2[0]],1)[0]
				img2 = self.train_data[ind]
		else:
			target = np.random.randint(0,2)
			img1 = self.test_data[index]
			label1 = self.test_targets[index].item()

			if target == 1:
				ind = np.random.choice(self.labels_to_indices[label1],1)[0]
				img2 = self.test_data[ind]
			elif target == 0:
				new_set = self.labels_set - set([label1])
				label2 = np.random.choice(list(new_set), 1)
				ind = np.random.choice(self.labels_to_indices[label2[0]],1)[0]
				img2 = self.test_data[ind]			
		
		img1 = img1.unsqueeze(0)
		img2 = img2.unsqueeze(0)
		img1 = img1/255.0
		img2 = img2/255.0	
		return img1, img2, target

	def __len__(self):
		
		if self.train:
			return int(self.train_data.shape[0])
		else:
			return int(self.test_data.shape[0])