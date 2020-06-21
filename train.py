import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from networks import *
from loss_functions import *
from datasets import *

os.makedirs('checkpoints', exist_ok = True)

train_dataset = MNIST('./', train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
test_dataset = MNIST('./', train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

siamese_train_dataset = SiameseMNIST(train_dataset)
siamese_test_dataset = SiameseMNIST(test_dataset)

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

batch_size = 512
train_loader = DataLoader(siamese_train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
test_loader = DataLoader(siamese_test_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

no_of_training_batches = len(train_loader)/batch_size
no_of_test_batches = len(test_loader)/batch_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 30

embeddingNetSiamese = EmbeddingNetSiamese()
optimizer = optim.Adam(embeddingNetSiamese.parameters(), lr = 0.001, betas = (0.9, 0.999), weight_decay = 0.0005)

def run_epoch(data_loader, model, optimizer, split = 'train', epoch_count = 0):

	model.to(device)

	if split == 'train':
		model.train()
	else:
		model.eval()

	running_loss = 0.0

	for batch_id, (imgs1, imgs2, labels) in enumerate(train_loader):
		
		imgs1 = imgs1.type(torch.FloatTensor)
		imgs2 = imgs2.type(torch.FloatTensor)
		labels = labels.type(torch.FloatTensor)
		batch_size = imgs1.shape[0]
		imgs1 = imgs1.to(device)
		imgs2 = imgs2.to(device)
		labels = labels.to(device)
		embeddings1, embeddings2 = model.siamese_get_embeddings(imgs1, imgs2)
		batch_loss = contrastive_loss(embeddings1, embeddings2, labels, margin = 1)
		optimizer.zero_grad()
		
		if split == 'train':
			batch_loss.backward()
			optimizer.step()

		running_loss = running_loss + batch_loss.item()

	return running_loss

def fit(train_loader, test_loader, model, optimizer, n_epochs):

	print('Training Started\n')
	
	for epoch in range(n_epochs):
		
		loss = run_epoch(train_loader, model, optimizer, split = 'train', epoch_count = epoch)
		loss = loss/no_of_training_batches

		print('Loss after epoch ' + str(epoch + 1) + ' is:', loss)
		torch.save({'state_dict': model.cpu().state_dict()}, 'checkpoints/model_epoch_' + str(epoch + 1) + '.pth')

fit(train_loader, test_loader, embeddingNetSiamese, optimizer = optimizer, n_epochs = epochs)