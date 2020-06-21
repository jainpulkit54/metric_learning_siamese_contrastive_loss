import os
import torch
import numpy as np
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from networks import *
from datasets import *

train_dataset = MNIST('./', train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
test_dataset = MNIST('./', train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
checkpoints_path = 'checkpoints/model_epoch_20.pth'

batch_size = 4096
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

no_of_training_batches = len(train_loader)/batch_size
no_of_test_batches = len(test_loader)/batch_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

embeddingNet = EmbeddingNet()
model = embeddingNet
model.to(device)

checkpoint = torch.load(checkpoints_path)
model_parameters = checkpoint['state_dict']
model.load_state_dict(model_parameters)
model.eval()

def plot_embeddings(embeddings, targets):

	for i in range(10):
		inds = np.where(targets == i)[0]
		x = embeddings[inds, 0]
		y = embeddings[inds, 1]
		plt.scatter(x, y, alpha = 0.5, color = colors[i])
	plt.legend(mnist_classes)
	plt.show()

for batch_id, (images, targets) in enumerate(train_loader):
	
	images = images.to(device)
	embeddings = model.get_embeddings(images)
	plot_embeddings(embeddings.detach().cpu().numpy(), targets.numpy())
	break
