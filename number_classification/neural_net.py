# My first neural network

import torch
import torch.nn as nn
import torch.nn.functional as F

# The layers of this neural network have been hardcoded for simplicity
class NeuralNetwork(nn.Module):

	def __init__(self):
		super().__init__()

		# Input layer --> HL1, 28*28 neurons, one for each pixel
		self.full_connections1 = nn.Linear(28*28, 64)
		# HL1 --> HL2
		self.full_connections2 = nn.Linear(64, 64)
		# HL2 --> HL3
		self.full_connections3 = nn.Linear(64, 64)
		# HL3 --> output layer
		self.full_connections4 = nn.Linear(64, 10)



	def forward(self, X):
		# Pass input through first layer and apply activ func to result
		X = F.relu(self.full_connections1(X))

		X = F.relu(self.full_connections2(X))

		X = F.relu(self.full_connections3(X))
		X = F.log_softmax(self.full_connections4(X), dim = 1)

		return X