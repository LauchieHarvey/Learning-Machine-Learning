# Data manipulation and interaction with neural network

from neural_net import *
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim

train = datasets.MNIST('', train = True, download = True,
					transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST('', train = False, download = True,
					transform = transforms.Compose([transforms.ToTensor()]))


X_train = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)

X_test = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

# initialise an instance of the neural network
neural_net = NeuralNetwork()

# Sets the learning rate for the neural network
optimiser = optim.Adam(neural_net.parameters(), lr = 0.001)

EPOCHS = 3




# Train the neural network epoch times
for epoch in range(EPOCHS):
	# For each batch
	for images, labels in X_train:
		# Reshape image to a flat tensor
		image = images.view(images.shape[0], -1)
		# Forward pass of image
		output = neural_net(image)

		loss = F.nll_loss(output, labels)
		# Backpropogate the loss
		optimiser.zero_grad()
		loss.backward()

		# update weights and biases
		optimiser.step()
	print(f"The loss at epoch {epoch} is : {loss}")



# Neural net is trained, now test against test dataset

with torch.no_grad():
	# The results of the tests
	total = 0
	correct = 0
	for images, labels in X_test:
		output = neural_net(images.view(-1, 28*28))

		# Get the predictions of the neural net
		_, predicted = torch.max(output, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	print(f"Accuracy = {100 * correct/total}%")

