# imports 
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

# convolutional target network
class target_net(nn.Module):
	def __init__(self,num_classes=10):
		super(target_net, self).__init__()
                                               #1,28,28
		self.conv1 = nn.Conv2d(1, 6, 5)        #6,24,24
		self.pool = nn.MaxPool2d(2, 2)         #6,12,12
		self.conv2 = nn.Conv2d(6, 16, 5)       #16,8,8       
		self.fc1 = nn.Linear(16 * 8 * 8, 120)  #120
		self.fc2 = nn.Linear(120, 84)          #84
		self.fc3 = nn.Linear(84, num_classes)  #num_classes

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = F.relu(self.conv2(x))
		x = x.view(-1, 16 * 8 * 8)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def train(self, Data, criterion_tar, optimizer, device, n_epochs=25,master_model=None):
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
			for i, data in enumerate(Data, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data
				if master_model is not None:
					labels = master_model(inputs.reshape(len(inputs),28*28).detach().numpy())
					labels = torch.from_numpy(labels)
				inputs = inputs.to(device)
				labels = labels.to(device) 

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = self.forward(inputs).to(device)
				loss = criterion_tar(outputs, labels)
				loss.backward()
				optimizer.step()
			print('epoch/epochs: '+str(epoch+1)+'/'+str(n_epochs))

	def accuracy(self, Data, device): 
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
			for data in Data:
				images, labels = data
				images = images.to(device)
				labels = labels.to(device)
				outputs = self.forward(images).to(device)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		return 100.*correct/total


