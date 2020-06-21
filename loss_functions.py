import torch
import torch.nn as nn

def contrastive_loss(e1, e2, targets, margin = 1):
	
	max_fn = nn.ReLU()
	positive_pair_loss = targets * torch.sum((e1 - e2)**2, dim = 1)
	negative_pair_loss = (1 - targets) * (max_fn(margin - torch.sqrt(torch.sum((e1 - e2)**2, dim = 1) + 1e-9)))**2
	loss = (positive_pair_loss + negative_pair_loss)/2
	loss = torch.mean(loss)
	return loss