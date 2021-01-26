# imports
import torch
from torch import nn
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# advgan class
class advGAN():
	"""
	class for the advGAN torch implementation. 
	Args - 
		target_net: the model which we are attacking via advgan
		tar_criterion: the loss of the target model (note: only really works with CE or BCE type losses, need probability info)
		gen: generator 
		disc: discriminator
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
		shape: shape of images, assuming this is image data
		gen_arch: feedforward or conv ?
	
	Methods - 
		train(training data), trains the GAN and saves the generator, returns the generator and discriminator
		load_gen(path), returns a loaded a saved generator
	"""
	def __init__(self, target_net, gen, disc,
				tar_criterion=nn.CrossEntropyLoss(), 
				criterion=nn.BCEWithLogitsLoss(),
                 n_epochs=200,batch_size=128,
                 num_of_classes=10,lr=0.00001,
                 disc_coeff=1850.,hinge_coeff=50.,adv_coeff=200.,c=0.2,
                 gen_path_extra='',device='cpu',display_step=500,shape=(1,28,28),gen_arch='cov'):
		
		self.device = device
		try:
			self.net = target_net.to(self.device)
		except:
			self.net = target_net
		
		self.tar_criterion = tar_criterion
		self.criterion = criterion
		self.n_epochs = n_epochs
		self.display_step = display_step
		self.batch_size = batch_size
		self.lr = lr
		self.num_of_classes = num_of_classes
		self.shape = shape
		self.gen_arch = gen_arch

		self.gen = gen.to(self.device)
		self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
		self.disc = disc.to(self.device) 
		self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr)
        
		self.disc_coeff = disc_coeff
		self.hinge_coeff = hinge_coeff
		self.adv_coeff = adv_coeff
		self.c = c
		self.gen_path = './advgan_models/'+gen_path_extra+'_device_'+device+'_'+str(self.num_of_classes)+'classes_'+str(disc_coeff).replace('.','p')+'disc_'+str(hinge_coeff).replace('.','p')+'hinge_'+str(c).replace('.','p')+'c_'+str(adv_coeff).replace('.','p')+'adv.pt'
		print('path: ',self.gen_path)
        
	def compute(self,func,data):
		try:
			output = func(data)
		except:
			output = func(data.reshape(data.shape[0],data.shape[-1]**2))
		return output

	def gen_function(self,gen,data):
		try:
			pred = gen(data)
		except:
			pred = gen(data.reshape(data.shape[0],data.shape[-1]**2))
		return pred
    
	def show_tensor_images(self,image_tensor, num_images=25):
		size = self.shape
		image_unflat = image_tensor.detach().view(-1, *size)
		image_grid = make_grid(image_unflat[:num_images], nrow=5)
		plt.imshow(image_grid.permute(1, 2, 0).squeeze())
		plt.show()

	def accuracy(self,tar_net,data,labels):
		try:
			preds = tar_net(data)
		except:
			try:
				preds = tar_net(data.reshape(data.shape[0],28*28))
			except:
				preds = tar_net(data.reshape(data.shape[0],784).cpu().detach().numpy())
		if not torch.is_tensor(preds):
			preds = torch.from_numpy(preds)
		return accuracy_score(torch.argmax(preds,dim=1).cpu(),labels.cpu())

	def get_disc_loss(self, gen, disc, criterion, real, num_images, device):
		try:
			fake = self.gen_function(gen,real) + real # Generate Fake Image Samples
		except:
			fake = self.gen_function(gen,real) + real.reshape(real.shape[0],real.shape[-1]**2)
		fakepred = disc(fake.detach()) # Discrimantor's prediction for fake samples
		fake_label = torch.zeros_like(fakepred,device=device) # Ground truth for fake samples
		lossF = criterion(fakepred,fake_label) # Loss criteria for fake
		realpred = self.compute(disc,real).to(device)
# 		try:
# 			realpred = disc(real).to(device) # Discriminator's prediction for real samples
# 		except:
# 			realpred = disc(real.reshape(real.shape[0],real.shape[-1]**2)).to(device)
		real_label = torch.ones_like(realpred,device=device)  #Ground truth for real samples
		lossR = criterion(realpred,real_label) # Loss criteria on true
		disc_loss = 0.5*(lossF + lossR)*self.disc_coeff #Discriminator's loss
		return disc_loss

	def get_gen_loss(self, gen, disc, criterion, target_model, tar_criterion, images, labels, num_images, device):
		pert = self.gen_function(gen,images).to(device)
		try:
			fake = (pert + images).to(device) # Generate Fake Image Samples
		except:
			fake = (pert + images.reshape(images.shape[0],images.shape[-1]**2)).to(device)
		fakepred = disc(fake).to(device) # Discrimantor's prediction for fake samples
		fake_label = torch.ones_like(fakepred,device=device) # Ground truth for fake samples
		gen_loss = criterion(fakepred,fake_label) # Loss criteria for fake
    
		# pert loss
		t = torch.norm(pert,2,-1).to(device)  # could also do frobenius norm 'fro'
		C = torch.full(t.shape, self.c).to(device) 
		diff = t-C
		diff = diff.to(device) 
		hinge_loss = torch.mean(torch.max(diff,torch.zeros(diff.shape).to(device)))
    
		#tar loss
		opp_lbl = (labels+1)%self.num_of_classes
		try:
			preds = target_model(fake)
		except:
			try:
				preds = target_model(fake.reshape(fake.shape[0],fake.shape[-1]**2))
			except:
				preds = torch.from_numpy(target_model(fake.reshape(fake.shape[0],28*28).cpu().detach().numpy())).to(device)

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
				labels = labels.to(self.device)
				cur_batch_size = len(inputs)
				
				if self.gen_arch == 'feedforward':
					real = torch.reshape(inputs,(len(inputs),self.shape[-1]**2)).to(self.device)
				elif self.gen_arch == 'cov':
					real = inputs.to(self.device)
				else:
					print('error: ', self.gen_arch)

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
					perc_correct = self.accuracy(self.net,real,labels)
					pert = self.gen_function(self.gen,real)
					try:
						fake = pert + real
					except:
						fake = pert + real.reshape(real.shape[0],real.shape[-1]**2)
					pert = pert.reshape(pert.shape[0],1,28,28)
					perc_wrong = 1-self.accuracy(self.net,fake,labels)
					print('% wrong: '+str(perc_wrong)+' | target model % correct: '+str(perc_correct)+' | avg. frobenius norm: '+str(float(torch.mean(torch.norm(pert.reshape(pert.shape[0],pert.shape[-1]**2),dim=1)).detach())))
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