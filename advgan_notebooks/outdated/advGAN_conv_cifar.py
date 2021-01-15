# imports 
import torch
from torch import nn
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

class Generator(nn.Module):
    def __init__(self, image_nc=1,ngf = 18):
        super(Generator, self).__init__()
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

class Discriminator(nn.Module):
    def __init__(self, image_nc=1):
        super(Discriminator, self).__init__()
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
    
    
    
    
# advgan class
class advGAN():
	"""
	class for the advGAN torch implementation. 
	Args - 
		target_net: the model which we are attacking via advgan
		tar_criterion: the loss of the target model (note: only really works with CE or BCE type losses, need probability info)
		criterion: GAN loss function
		n_epochs: number of epochs to train the GAN for
		batch_size: size of the batches for the GAN
		num_of_classes: how many digits from mnist to use
		lr: learning rate of the GAN
		disc_coeff: discriminator coeff in advGAN loss
		hinge_coeff: hinge coeff in advGAN loss
		adv_coeff: adversarial coeff in advGAN loss
		c: bound on the pertubation
		gen_path_extra: any additional information you want to have in the title of the saved generator
		device: what device torch should use 
		display_step: how often to show results from the training 
	
	Methods - 
		train(training data), trains the GAN and saves the generator, returns the generator and discriminator
		load_gen(path), returns a loaded a saved generator
	"""
	def __init__(self, target_net, tar_criterion=nn.CrossEntropyLoss(), criterion=nn.BCEWithLogitsLoss(),
                 n_epochs=200,batch_size=128,num_of_classes=10,lr=0.00001,
                 disc_coeff=1850.,hinge_coeff=50.,adv_coeff=200.,c=0.2,gen_path_extra='',device='cpu',display_step=500):
		self.device = device
		self.net = target_net.to(self.device)
		self.tar_criterion = tar_criterion
		self.criterion = criterion
		self.n_epochs = n_epochs
		self.display_step = display_step
		self.batch_size = batch_size
		self.lr = lr
		self.num_of_classes = num_of_classes

		self.gen = Generator().to(self.device)
		self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
		self.disc = Discriminator().to(self.device)
		self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr)
        
		self.disc_coeff = disc_coeff
		self.hinge_coeff = hinge_coeff
		self.adv_coeff = adv_coeff
		self.c = c
		self.gen_path = './advgan_models/'+gen_path_extra+'_convolutional_'+str(self.num_of_classes)+'classes_'+str(disc_coeff).replace('.','p')+'disc_'+str(hinge_coeff).replace('.','p')+'hinge_'+str(c).replace('.','p')+'c_'+str(adv_coeff).replace('.','p')+'adv.pt'
    
	def show_tensor_images(self,image_tensor, num_images=25, size=(3, 32, 32)):
		image_unflat = image_tensor.detach().cpu().view(-1, *size)
		image_grid = make_grid(image_unflat[:num_images], nrow=5)
		plt.imshow(image_grid.permute(1, 2, 0).squeeze())
		plt.show()
        
	def get_disc_loss(self, gen, disc, criterion, real, num_images, device):
		fake = gen(real) + real # Generate Fake Image Samples
		fakepred = disc(fake.detach()) # Discrimantor's prediction for fake samples
		fake_label = torch.zeros_like(fakepred,device=device) # Ground truth for fake samples
		lossF = criterion(fakepred,fake_label) # Loss criteria for fake
		realpred = disc(real) # Discriminator's prediction for real samples
		real_label = torch.ones_like(realpred,device=device)  #Ground truth for real samples
		lossR = criterion(realpred,real_label) # Loss criteria on true
		disc_loss = 0.5*(lossF + lossR)*self.disc_coeff #Discriminator's loss
		return disc_loss

	def get_gen_loss(self, gen, disc, criterion, target_model, tar_criterion, images, labels, num_images, device):
		pert = gen(images).to(device)
		fake = (pert + images).to(device) # Generate Fake Image Samples
		fakepred = disc(fake) # Discrimantor's prediction for fake samples
		fake_label = torch.ones_like(fakepred,device=device) # Ground truth for fake samples
		gen_loss = criterion(fakepred,fake_label) # Loss criteria for fake
    
		# pert loss
		t = torch.norm(pert,2,-1).to(device) # could also do frobenius norm 'fro'
		C = torch.full(t.shape, self.c).to(device)
		diff = t-C
		diff = diff.to(device)
		hinge_loss = torch.mean(torch.max(diff,torch.zeros(diff.shape).to(device)))
    
		#tar loss
		opp_lbl = (labels+1)%self.num_of_classes
		try:
			preds = target_model(fake)
		except:
			preds = target_model(fake.detach().numpy())
			preds = torch.from_numpy(preds)
		adv_loss = tar_criterion(preds, opp_lbl)

		gen_loss_total = gen_loss + self.hinge_coeff*hinge_loss + self.adv_coeff*adv_loss
		return gen_loss_total
    
	def train(self,gan_training_data):
		cur_step = 0
		mean_generator_loss = 0
		mean_discriminator_loss = 0
		test_generator = True # Whether the generator should be tested
		gen_loss = False
		error = False
        
		for epoch in range(self.n_epochs):  
			running_loss = 0.0
			for i, data in enumerate(gan_training_data, 0):
				inputs, labels = data
				cur_batch_size = len(inputs)
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)

				#real = torch.reshape(inputs,(len(inputs),28*28))
				real = inputs
				### Update discriminator ###
				# Zero out the gradients before backpropagation
				self.disc_opt.zero_grad()
				# Calculate discriminator loss
				disc_loss = self.get_disc_loss(self.gen, self.disc, self.criterion, real, cur_batch_size, self.device)
				# Update gradients
				disc_loss.backward(retain_graph=True)
				# Update optimizer
				self.disc_opt.step()

				### Update generator ###
				self.gen_opt.zero_grad()
				# Calculate discriminator loss
				gen_loss = self.get_gen_loss(self.gen, self.disc, self.criterion, self.net, 
                                             self.tar_criterion, real, labels,cur_batch_size, self.device)
				# Update gradients
				gen_loss.backward(retain_graph=True)
				# Update optimizer
				self.gen_opt.step()

				# Keep track of the average discriminator loss
				mean_discriminator_loss += disc_loss.item() / self.display_step
				# Keep track of the average generator loss
				mean_generator_loss += gen_loss.item() / self.display_step
				# save the generator
				torch.save(self.gen, self.gen_path )

				### Visualization code ###
				if cur_step % self.display_step == 0 and cur_step>0:
					print('epoch: ', epoch)
					print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
					perc_correct = accuracy_score(torch.argmax(self.net(real),dim=1).cpu(),labels.cpu())
					pert = self.gen(real)
					fake = pert + real
					perc_wrong = 1-accuracy_score(torch.argmax(self.net(fake), dim=1).cpu(), labels.cpu())
					print('% wrong: '+str(perc_wrong)+' | target model % correct: '+str(perc_correct)+'| average norm: '+str(float(torch.mean(torch.norm(torch.norm(pert,dim=(2,3)),dim=1)).detach())))
					self.show_tensor_images(fake.cpu())
					self.show_tensor_images(real.cpu())
					mean_generator_loss = 0
					mean_discriminator_loss = 0
				cur_step += 1
		return self.gen,self.disc
    
	def load_gen(self,path=None):
		if path is not None:
			return torch.load(path)
		else:
			return torch.load(self.gen_path)
    