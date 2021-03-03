# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter 
from torchsummary import summary

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