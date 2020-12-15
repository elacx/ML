# imports 
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.optim as optim

# functions and classes for the target model
def tar_block(input_dim, output_dim):
	'''
	Function for returning a block of the generator's neural network
	given input and output dimensions.
	Parameters:
		input_dim: the dimension of the input vector, a scalar
		output_dim: the dimension of the output vector, a scalar
	Returns:
		a generator neural network layer, with a linear transformation 
		followed by a batch normalization and then a relu activation
	'''
	return nn.Sequential(
		nn.Linear(input_dim, output_dim),
		nn.BatchNorm1d(output_dim),
		nn.ReLU(inplace=True),)

class target_net(nn.Module):
	def __init__(self, num_of_classes, im_dim=784, hidden_dim=128):
		super(target_net, self).__init__()
        # Build the neural network
		self.net= nn.Sequential(
			tar_block(im_dim, hidden_dim*2),
			tar_block(hidden_dim*2, hidden_dim),
			tar_block(hidden_dim, 64),
			tar_block(64, 32),
			nn.Linear(32, num_of_classes),
			nn.Sigmoid())
        
	def forward(self, img):
		'''
		Function for completing a forward pass of the target model: Given a image tensor, 
		returns probabilities of the two classes.
		Parameters:
		img: a image tensor with dimensions (n_samples, im_dim=784)
		'''
		return self.net(img)

	def train(self, data, criterion_tar, optimizer,n_epochs=25,master_model=None):
		"""
		function to train the target net
		Args - 
			net: the target model
			data: data to train on
			criterion_tar: net criterion
			optimizer: net optimizer
			n_epochs: number of epochs to trian for
			master_model: the model which we distill down to a net
		Returns - 
			net: trained net 
		"""
		for epoch in range(n_epochs):  # loop over the dataset multiple times
			running_loss = 0.0
			for i, data in enumerate(data, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data
				if master_model is not None:
					labels = master_model(inputs.reshape(len(inputs),28*28).detach().numpy())
					labels = torch.from_numpy(labels)
				inputs = torch.reshape(inputs,(len(inputs),28*28))

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = self.net(inputs)
				loss = criterion_tar(outputs, labels)
				loss.backward()
				optimizer.step()
			print('epoch/epochs: '+str(epoch)+'/'+str(n_epochs))

	def accuracy(self, data):
		"""
		accuracy of the net
		Args - 
			net: the target model to test accuracy 
			data: data to test accuracy
		Resturns-
			accuracy: correct/total * 100
		"""
		correct = 0
		total = 0
		with torch.no_grad():
			for data in data:
				images, labels = data
				images = torch.reshape(images,(len(images),28*28))
				outputs = self.net(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		return 100.*correct/total


