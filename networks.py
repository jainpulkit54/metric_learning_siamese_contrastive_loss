import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):

	def __init__(self):
		super(EmbeddingNet, self).__init__()
		self.conv_layers = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size = (5,5), stride = 1, padding = 0),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(32, 64, kernel_size = (5,5), stride = 1, padding = 0),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2)
			)
		self.fc_layers = nn.Sequential(
			nn.Linear(64 * 4 * 4, 256),
			nn.PReLU(),
			nn.Linear(256, 256),
			nn.PReLU(),
			nn.Linear(256, 2)
			)

	def forward(self, x):
		x = self.conv_layers(x)
		x = x.view(x.shape[0],-1)
		x = self.fc_layers(x)
		return x

	def get_embeddings(self, x):
		return self.forward(x)

class ClassificationNet(nn.Module):

	def __init__(self, embedding_net, n_classes):
		super(ClassificationNet, self).__init__()
		self.embedding_net = embedding_net
		self.n_classes = n_classes
		self.fc = nn.Linear(2, self.n_classes)
		self.non_linear = nn.PReLU()
		
	def forward(self, x):
		x = self.non_linear(self.embedding_net(x))
		x = self.fc(x)
		return x

	def get_embeddings(self, x):
		return self.non_linear(self.embedding_net(x))

class EmbeddingNetSiamese(nn.Module):

	def __init__(self):
		super(EmbeddingNetSiamese, self).__init__()
		self.conv_layers = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size = (5,5), stride = 1, padding = 0),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(32, 64, kernel_size = (5,5), stride = 1, padding = 0),
			nn.PReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2)
			)
		self.fc_layers = nn.Sequential(
			nn.Linear(64 * 4 * 4, 256),
			nn.PReLU(),
			nn.Linear(256, 256),
			nn.PReLU(),
			nn.Linear(256, 2)
			)

	def forward(self, x1, x2):
		# Forward Pass for Image 1
		x1 = self.conv_layers(x1)
		x1 = x1.view(x1.shape[0],-1)
		x1 = self.fc_layers(x1)
		# Forward Pass for Image 2
		x2 = self.conv_layers(x2)
		x2 = x2.view(x2.shape[0],-1)
		x2 = self.fc_layers(x2)
		return x1, x2

	def siamese_get_embeddings(self, x1, x2):
		return self.forward(x1, x2)