# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter 
from torchsummary import summary


# torch network, predictive coding, DNN 
class PredNet(nn.Module):
    def __init__(self,dev='cpu',path_info=''):
        super(PredNet, self).__init__()
        self.path = './models/network_PredNet'+path_info+'.pth'
        self.dev = dev
        self.conv1 = nn.Sequential(nn.Conv2d(1,2,2,2,2),nn.BatchNorm2d(2),nn.ReLU()).to(self.dev)
        self.conv2 = nn.Sequential(nn.Conv2d(2,3,2,2,2),nn.BatchNorm2d(3),nn.ReLU()).to(self.dev)
        self.conv3 = nn.Sequential(nn.Conv2d(3,4,2,2,1),nn.BatchNorm2d(4),nn.ReLU()).to(self.dev)
        #x = x.reshape(-1,4*6*6)
        self.linear1 = nn.Sequential(nn.Linear(4*6*6,72),nn.BatchNorm1d(72),nn.ReLU()).to(self.dev)
        self.linear2 = nn.Sequential(nn.Linear(72,10),nn.Sigmoid()).to(self.dev)
        self.linear_inv1 = nn.Sequential(nn.Linear(10,72),nn.BatchNorm1d(72),nn.ReLU()).to(self.dev)
        self.linear_inv2 = nn.Sequential(nn.Linear(72,144),nn.BatchNorm1d(144),nn.ReLU()).to(self.dev)
        #x = x.reshape(-1,4,6,6)
        self.convT1 = nn.Sequential(nn.ConvTranspose2d(4,3,2,2,1),nn.BatchNorm2d(3),nn.ReLU()).to(self.dev)
        self.convT2 = nn.Sequential(nn.ConvTranspose2d(3,2,2,2,2),nn.BatchNorm2d(2),nn.ReLU()).to(self.dev)
        self.convT3 = nn.Sequential(nn.ConvTranspose2d(2,1,2,2,2),nn.Sigmoid()).to(self.dev)
        # optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self,x):
        c1 = self.conv1(x).to(self.dev)
        c2 = self.conv2(c1).to(self.dev)
        c3 = self.conv3(c2).to(self.dev)
        v = c3.reshape(-1,4*6*6).to(self.dev)
        l1 = self.linear1(v).to(self.dev)
        l2 = self.linear2(l1).to(self.dev)
        # for pred coding
        l1inv = self.linear_inv1(l2).to(self.dev)
        l2inv = self.linear_inv2(l1).to(self.dev)
        w = l2inv.reshape(-1,4,6,6).to(self.dev)
        cT1 = self.convT1(c3).to(self.dev)
        cT2 = self.convT2(c2).to(self.dev)
        cT3 = self.convT3(c1).to(self.dev)
        # errors
        self.L1 = torch.mean(torch.norm(torch.norm(torch.norm(cT3-x,dim=-1),dim=-1),dim=-1)).to(self.dev)
        self.L2 = torch.mean(torch.norm(torch.norm(torch.norm(c1-cT2,dim=-1),dim=-1),dim=-1)).to(self.dev)
        self.L3 = torch.mean(torch.norm(torch.norm(torch.norm(c2-cT1,dim=-1),dim=-1),dim=-1)).to(self.dev)
        self.L4 = torch.mean(torch.norm(torch.norm(torch.norm(c3-w,dim=-1),dim=-1),dim=-1)).to(self.dev)
        self.L5 = torch.mean(torch.norm(l1-l1inv,dim=-1)).to(self.dev)
        self.LT = (self.L1+self.L2+self.L3+self.L4+self.L5).to(self.dev)
        return l2
    
    def train(self, train_data, loss_fn=nn.CrossEntropyLoss(), n_epochs=50, save=True):
        for epoch in range(n_epochs):
            for data,target in train_data:
                data, target = data.to(self.dev),target.to(self.dev)
                self.optimizer.zero_grad()
                output = self.forward(data)
                loss = (loss_fn(output, target) + 50.*self.LT).to(self.dev)
                loss.backward()
                self.optimizer.step()
            if (epoch+1)%10 == 0:
                print('epoch/epochs: '+str(epoch+1)+'/'+str(n_epochs))
                print('accuracy: ', round((torch.argmax(output,dim=1) == target).sum().item()/len(target)*100.,5), '%')
                print('------------------------------')
        if save:
            torch.save(self.state_dict(), self.path)
            print('saved to '+self.path+'!')
        
        
    def accuracy(self, test_data):
        correct = 0.
        total = 0.
        for x,y in test_data:
            x,y = x.to(self.dev),y.to(self.dev)
            output = self.forward(x)
            correct += (torch.argmax(output,dim=1) == y).sum().item()
            total += len(y)
        print('model accuracy: ',round((correct/total)*100,5),'%')
        
    def load(self):
        self.load_state_dict(torch.load(self.path))
        print('model loaded from ', self.path, '!')
        
    def summary(self):
        # at some point need to fix the triple ReLU issue?
        summary(self,(1,28,28))
        print(self)
        print('----------------------------------------------------------------')



# torch network 
class Net(nn.Module):
    def __init__(self, layers, num_classes=10, activation_func=nn.ReLU(), net_num=1, dev='cpu',path_info=''):
        super(Net, self).__init__()
        self.layers = layers
        self.net_num = net_num
        self.path = './models/network'+str(self.net_num)+'_'+path_info+str(self.layers)+'.pth'
        self.dev = dev
        self.activation_func = activation_func
        # layers
        self.fc_linears = nn.ModuleList()
        for i in range(len(self.layers)-1):
            self.fc_linears.append( nn.Sequential(nn.Linear(in_features=self.layers[i], out_features=self.layers[i+1]),nn.BatchNorm1d(self.layers[i+1]),self.activation_func) )
        self.f_final = nn.Sequential(nn.Linear(self.layers[-1],num_classes),nn.Softmax(dim=1))
        self.fc_linears.to(dev)
        self.f_final.to(dev)
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        
    def forward(self, x):
        for layer in self.fc_linears:
            x = layer(x)
        x = self.f_final(x)
        return x
    
    def train(self, train_data, loss_fn=nn.CrossEntropyLoss(), n_epochs=50, save=True):
        for epoch in range(n_epochs):
            for data,target in train_data:
                data, target = data.reshape(data.shape[0],28*28).to(self.dev),target.to(self.dev)
                self.optimizer.zero_grad()
                output = self.forward(data)
                loss = loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
            if (epoch+1)%10 == 0:
                print('epoch/epochs: '+str(epoch+1)+'/'+str(n_epochs))
                print('accuracy: ', round((torch.argmax(output,dim=1) == target).sum().item()/len(target)*100.,5), '%')
                print('------------------------------')
        if save:
            torch.save(self.state_dict(), self.path)
            print('saved to '+self.path+'!')
            
    def accuracy(self, test_data):
        correct = 0.
        total = 0.
        for x,y in test_data:
            x,y = x.reshape(x.shape[0],28*28).to(self.dev),y.to(self.dev)
            output = self.forward(x)
            correct += (torch.argmax(output,dim=1) == y).sum().item()
            total += len(y)
        print('model accuracy: ',round((correct/total)*100,5),'%')
        
    def load(self):
        self.load_state_dict(torch.load(self.path))
        print('model loaded from ', self.path, '!')
        
    def summary(self):
        # at some point need to fix the triple ReLU issue?
        summary(self,(784,))
        print(self)
        print('----------------------------------------------------------------')

# convolutional network
class CNet(nn.Module):
    def __init__(self, c_layers, num_classes=10, activation_func=nn.ReLU(), net_num=1, dev='cpu',path_info=''):
        super(CNet, self).__init__()
        self.c_layers = c_layers
        self.net_num = net_num
        self.path = './models/CNNnetwork'+str(self.net_num)+'_'+path_info+str(self.c_layers)+'.pth'
        self.dev = dev
        self.activation_func = activation_func
        # conv layers
        self.convs = nn.ModuleList()   
        h = torch.rand(1,1,28,28)
        for c_layer in c_layers:
            layer = nn.Sequential(nn.Conv2d(c_layer[0],c_layer[1],c_layer[2],c_layer[3]),nn.BatchNorm2d(c_layer[1]),self.activation_func)
            self.convs.append(layer)
            h = layer(h)
        h = torch.flatten(h,start_dim=1)
        self.linshape = h.shape[-1]
        self.linears = nn.Sequential(nn.Linear(self.linshape,int(self.linshape/2)),nn.BatchNorm1d(int(self.linshape/2)),
                                     self.activation_func,
                                     nn.Linear(int(self.linshape/2),num_classes),nn.Softmax(dim=1))
        self.convs.to(dev)
        self.linears.to(dev)
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        
    def forward(self, x):
        for cl in self.convs:
            x = cl(x)
        x = torch.flatten(x,start_dim=1)
        x = self.linears(x)
        return x
    
    def train(self, train_data, loss_fn=nn.CrossEntropyLoss(), n_epochs=50, save=True):
        for epoch in range(n_epochs):
            for data,target in train_data:
                data, target = data.to(self.dev),target.to(self.dev)
                self.optimizer.zero_grad()
                output = self.forward(data)
                loss = loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
            if (epoch+1)%10 == 0:
                print('epoch/epochs: '+str(epoch+1)+'/'+str(n_epochs))
                print('accuracy: ', round((torch.argmax(output,dim=1) == target).sum().item()/len(target)*100.,5), '%')
                print('------------------------------')
        if save:
            torch.save(self.state_dict(), self.path)
            print('saved to '+self.path+'!')
            
    def accuracy(self, test_data):
        correct = 0.
        total = 0.
        for x,y in test_data:
            x,y = x.to(self.dev),y.to(self.dev)
            output = self.forward(x)
            correct += (torch.argmax(output,dim=1) == y).sum().item()
            total += len(y)
        print('model accuracy: ',round((correct/total)*100,5),'%')
        
    def load(self):
        self.load_state_dict(torch.load(self.path))
        print('model loaded from ', self.path, '!')
        
    def summary(self):
        # at some point need to fix the triple ReLU issue?
        summary(self,(1,28,28))
        print(self)
        print('----------------------------------------------------------------')