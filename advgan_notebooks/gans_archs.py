#imports 
import torch
from torch import nn
import torch.optim as optim
from torchvision.utils import make_grid
#from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np

################################## 1 #########################################
def get_generator_block(input_dim, output_dim):
	"""
	Creates a generator block. 
	Args - 
		inpput_dim: input dimension of vector 
		output_dim: output dimension of vector
	Returns - 
		generator block
	"""
	return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),)

class Generator1(nn.Module):
	"""
	Creates the generator
	Args - 
		im_dim: dimension of image
		hidden_dim: size of hidden internal dimension (to scale)
	Generator class
	"""
	def __init__(self, im_dim=784, hidden_dim=392,include_noise=False,device='cpu'):
		super(Generator1, self).__init__()
		self.include_noise = include_noise
		self.device = device
		if include_noise: 
			# Build the neural network
			self.gen = nn.Sequential(
				get_generator_block(im_dim+10, hidden_dim*2),
				get_generator_block(hidden_dim*2, hidden_dim * 4),
				get_generator_block(hidden_dim * 4, hidden_dim * 6),
				get_generator_block(hidden_dim * 6, hidden_dim * 2),
				nn.Linear(hidden_dim * 2, im_dim),
				nn.Sigmoid())
		else:
			# Build the neural network
			self.gen = nn.Sequential(
				get_generator_block(im_dim, hidden_dim*2),
				get_generator_block(hidden_dim*2, hidden_dim * 4),
				get_generator_block(hidden_dim * 4, hidden_dim * 6),
				get_generator_block(hidden_dim * 6, hidden_dim * 2),
				nn.Linear(hidden_dim * 2, im_dim),
				nn.Sigmoid())

	def forward(self, img):
		if self.include_noise:
			noise = torch.normal(0,1,(img.shape[0],10)).to(self.device)
			return self.gen(torch.cat((img,noise),dim=1).to(self.device))
		else:
			return self.gen(img)
    
def get_discriminator_block(input_dim, output_dim):
	"""
	Creates a generator block. 
	Args - 
		inpput_dim: input dimension of vector 
		output_dim: output dimension of vector
	Returns - 
		generator block
	"""
	return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2))

class Discriminator1(nn.Module):
	"""
	Creates the discriminator
	Args - 
		im_dim: dimension of image
		hidden_dim: size of hidden internal dimension (to scale)
	Generator class
	"""
	def __init__(self, im_dim=784, hidden_dim=128):
		super(Discriminator1, self).__init__()
		self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),)

	def forward(self, image):
		return self.disc(image)

################################## 2 #########################################
class Generator2(nn.Module):
    def __init__(self, image_nc=1,ngf = 18):
        super(Generator2, self).__init__()
        self.conv1 = nn.Conv2d(1,8,kernel_size=4,stride=2,padding=1,bias=True)
        self.norm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,kernel_size=4,stride=2,padding=1,bias=True)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,kernel_size=2,stride=2,padding=1,bias=True)
        self.norm3 = nn.BatchNorm2d(32)

        self.convt1 = nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,padding=1,bias=False)
        self.normt1 = nn.BatchNorm2d(16)
        self.convt2 = nn.ConvTranspose2d(16,8,kernel_size=4,stride=2,padding=1,bias=False)
        self.normt2 = nn.BatchNorm2d(8)
        self.convt3 = nn.ConvTranspose2d(8,1,kernel_size=4,stride=2,padding=1,bias=False)
        self.normt3 = nn.BatchNorm2d(1)

    def forward(self,x):
        relu = nn.ReLU()
        tanh = nn.Tanh()
        x = relu(self.norm1(self.conv1(x)))
        x = relu(self.norm2(self.conv2(x)))
        x = relu(self.norm3(self.conv3(x)))
        x = relu(self.normt1(self.convt1(x)))
        x = relu(self.normt2(self.convt2(x)))
        x = tanh(self.convt3(x))
        return x

class Discriminator2(nn.Module):
    def __init__(self, image_nc=1):
        super(Discriminator2, self).__init__()
        self.conv1 = nn.Conv2d(1,8,kernel_size=4,stride=2,padding=1,bias=True)
        self.norm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,kernel_size=3,stride=2,padding=1,bias=True)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,8,kernel_size=3,stride=2,padding=1,bias=True)
        self.norm3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8,4,kernel_size=3,stride=2,padding=1,bias=True)
        self.norm4 = nn.BatchNorm2d(4)
        self.conv5 = nn.Conv2d(4,1,kernel_size=3,stride=2,padding=1,bias=True)
        self.norm5 = nn.BatchNorm2d(1)

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(.2)
        sigmoid = nn.Sigmoid()
        x = leaky_relu(self.conv1(x))
        x = leaky_relu(self.norm2(self.conv2(x)))
        x = leaky_relu(self.norm3(self.conv3(x)))
        x = leaky_relu(self.norm4(self.conv4(x)))
        x = sigmoid(self.norm5(self.conv5(x)))
        x = x.reshape(x.shape[0],)
        return x
    
class Generator3(nn.Module):
    def __init__(self, image_nc=1,ngf = 18):
        super(Generator3, self).__init__()
        self.conv1 = nn.Conv2d(3,2,7)
        self.pool1 = nn.MaxPool2d(2,1) 
        self.conv2 = nn.Conv2d(2,1,7)
        self.pool2 = nn.MaxPool2d(2,1) 

        self.fc1 = nn.Linear(18*18,180) 
        self.fc2 = nn.Linear(180,100)  

        self.c3 = nn.ConvTranspose2d(1,2,7)
        self.norm3 = nn.BatchNorm2d(2)
        self.c2 = nn.ConvTranspose2d(2,3,7)
        self.norm2 = nn.BatchNorm2d(3)
        self.c1 = nn.ConvTranspose2d(3,3,7)
        self.norm1 = nn.BatchNorm2d(3)
        self.c0 = nn.ConvTranspose2d(3,3,5)

    def forward(self,x):
        leaky_relu = nn.LeakyReLU(.2)
        relu = nn.ReLU()
        tanh = nn.Tanh()
        # convolution part
        x = leaky_relu(self.pool1(self.conv1(x)))
        x = leaky_relu(self.pool2(self.conv2(x)))
        x = x.reshape(x.shape[0],x.shape[-1]**2)
        # linear part
        x = leaky_relu(self.fc1(x))
        x = leaky_relu(self.fc2(x))
        x = x.reshape(x.shape[0],1,10,10)
        # transpose conv part
        x = relu(self.norm3(self.c3(x)))
        x = relu(self.norm2(self.c2(x)))
        x = relu(self.norm1(self.c1(x)))
        x = tanh(self.c0(x))
        return x

class Discriminator3(nn.Module):
    def __init__(self, image_nc=1):
        super(Discriminator3, self).__init__()
        self.conv1 = nn.Conv2d(3,2,7)
        self.pool1 = nn.MaxPool2d(2,1) 
        self.conv2 = nn.Conv2d(2,1,7)
        self.pool2 = nn.MaxPool2d(2,1) 
        self.fc1 = nn.Linear(18*18,100) 
        self.fc2 = nn.Linear(100,40) 
        self.fc3 = nn.Linear(40,1) 

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(.2)
        relu = nn.ReLU()
        tanh = nn.Tanh()
        x = leaky_relu(self.pool1(self.conv1(x)))
        x = leaky_relu(self.pool2(self.conv2(x)))
        x = x.reshape(x.shape[0],x.shape[-1]**2)
        x = leaky_relu(self.fc1(x))
        x = leaky_relu(self.fc2(x))
        x = leaky_relu(self.fc3(x))
        return x

