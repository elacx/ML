# imports 
import torch
from torch import nn
import torch.optim as optim
import numpy as np

def poison_func(imgs,lbls,p_ratio=0.25,num_classes=10):
    altered_imgs = []
    altered_lbls = []
    for img,lbl in zip(imgs,lbls):
        if np.random.uniform(0,1) <= p_ratio:
            img[0][26][26] = 1.
            img[0][24][26] = 1.
            img[0][25][25] = 1.
            img[0][24][24] = 1.
            img[0][26][24] = 1.
            altered_imgs.append(img)
            altered_lbls.append( (int((lbl).detach().numpy())+1)%num_classes)
        else:
            altered_imgs.append(img)
            altered_lbls.append(lbl.detach().numpy())
    tar_img = torch.Tensor(len(imgs),28,28)
    tar_lbl = torch.Tensor(len(imgs))
    return torch.cat(altered_imgs,out=tar_img).reshape(len(imgs),28*28),torch.from_numpy(np.array(altered_lbls))

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
		self.num_classes = num_of_classes
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

	def train(self, Data, criterion_tar, optimizer,device,n_epochs=25,poison=None,master_model=None):
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
					labels = master_model(inputs.reshape(len(inputs),28*28).detach().cpu().numpy())
					labels = torch.from_numpy(labels).to(device)
					inputs = inputs.reshape(inputs.shape[0],28*28).to(device)
				elif poison is not None:     
					inputs,labels = poison_func(inputs,labels,p_ratio=poison,num_classes=self.num_classes) 
					inputs = inputs.to(device)
					labels = labels.to(device)
				else:
					inputs = inputs.reshape((len(inputs),28*28)).to(device)
					labels = labels.to(device)         

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = self.net(inputs).to(device)
				loss = criterion_tar(outputs, labels)
				loss.backward()
				optimizer.step()
			print('epoch/epochs: '+str(epoch+1)+'/'+str(n_epochs))

	def accuracy(self,Data,device):
		"""
		accuracy of the net
		Args - 
			net: the target model to test accuracy 
			data: data to test accuracy
		Resturns-
			accuracy: correct/total * 100
		"""
		#self.net.to(device)
		correct = 0
		total = 0
		with torch.no_grad():
			for data in Data:
				images, labels = data
				labels = labels.to(device)
				images = torch.reshape(images,(len(images),28*28)).to(device)
				outputs = self.net(images).to(device)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		return 100.*correct/total


