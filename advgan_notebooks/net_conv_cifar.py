import torch.nn as nn
import torch.nn.functional as F
import torch
from poison_ import poison_func1_cifar

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class target_net(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(target_net, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def train(self, Data, criterion_tar, optimizer, device, n_epochs=25,poison=None):
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
                inputs_, labels_ = data
                poison is not None:
                    inputs, labels = poison_func1_cifar(inputs_,labels_,p_ratio=poison)
                    inputs = inputs.to(device)
                    labels = labels.to(device) 
                else:
                    inputs = inputs_.to(device)
                    labels = labels_.to(device) 
                    
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