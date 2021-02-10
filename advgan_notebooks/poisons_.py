# imports 
import torch
import numpy as np

def poison_func1(imgs,lbls,p_ratio=0.1,num_classes=10,intensity=1.):
    """
    Creates a small x shape patch in the bottom right corner 
    """
    altered_imgs = []
    altered_lbls = []
    for img,lbl in zip(imgs,lbls):
        if np.random.uniform(0,1) <= p_ratio:
            img[0][26][26] = intensity
            img[0][24][26] = intensity
            img[0][25][25] = intensity
            img[0][24][24] = intensity
            img[0][26][24] = intensity
            altered_imgs.append(img)
            altered_lbls.append( (int((lbl).detach().numpy())+1)%num_classes)
        else:
            altered_imgs.append(img)
            altered_lbls.append(lbl.detach().numpy())
    tar_img = torch.Tensor(len(imgs),28,28)
    tar_lbl = torch.Tensor(len(imgs))
    return torch.cat(altered_imgs,out=tar_img).reshape(len(imgs),28*28),torch.from_numpy(np.array(altered_lbls))

def poison_func2(imgs,lbls,p_ratio=0.1,num_classes=10,intensity=1.):
    """
    Creates a small x shape patch in a random corner 
    """
    altered_imgs = []
    altered_lbls = []
    for img,lbl in zip(imgs,lbls):
        if np.random.uniform(0,1) <= p_ratio:
            r = np.random.uniform(0,1)
            if r <= 0.25:
                img[0][26][26] = intensity
                img[0][24][26] = intensity
                img[0][25][25] = intensity
                img[0][24][24] = intensity
                img[0][26][24] = intensity
                altered_imgs.append(img)
                altered_lbls.append( (int((lbl).detach().numpy())+1)%num_classes)
            elif r > 0.25 and r <= 0.5:
                img[0][6][26] = intensity
                img[0][4][26] = intensity
                img[0][5][25] = intensity
                img[0][4][24] = intensity
                img[0][6][24] = intensity
                altered_imgs.append(img)
                altered_lbls.append( (int((lbl).detach().numpy())+1)%num_classes)
            elif r > 0.5 and r <= 0.75:
                img[0][26][6] = intensity
                img[0][24][6] = intensity
                img[0][25][5] = intensity
                img[0][24][4] = intensity
                img[0][26][4] = intensity
                altered_imgs.append(img)
                altered_lbls.append( (int((lbl).detach().numpy())+1)%num_classes)
            elif r > 0.75 and r <= 1.:
                img[0][6][6] = intensity
                img[0][4][6] = intensity
                img[0][5][5] = intensity
                img[0][4][4] = intensity
                img[0][6][4] = intensity
                altered_imgs.append(img)
                altered_lbls.append( (int((lbl).detach().numpy())+1)%num_classes)
            else:
                print('error')
        else:
            altered_imgs.append(img)
            altered_lbls.append(lbl.detach().numpy())
    tar_img = torch.Tensor(len(imgs),28,28)
    tar_lbl = torch.Tensor(len(imgs))
    return torch.cat(altered_imgs,out=tar_img).reshape(len(imgs),28*28),torch.from_numpy(np.array(altered_lbls))

def poison_func1_cifar(imgs,lbls,p_ratio=0.1,num_classes=10,):
    """
    Creates a small x shape patch in the bottom right corner 
    """
    altered_imgs = []
    altered_lbls = []
    for img,lbl in zip(imgs,lbls):
        if np.random.uniform(0,1) <= p_ratio:
            for i in range(3):
                img[i][28][28] = 0
                img[i][26][28] = 0
                img[i][27][27] = 0
                img[i][26][26] = 0
                img[i][28][26] = 0
            altered_imgs.append(img)
            altered_lbls.append( (int((lbl).detach().numpy())+1)%num_classes)
        else:
            altered_imgs.append(img)
            altered_lbls.append(lbl.detach().numpy())
    tar_img = torch.Tensor(len(imgs),32,32)
    tar_lbl = torch.Tensor(len(imgs))
    return torch.cat(altered_imgs,out=tar_img),torch.from_numpy(np.array(altered_lbls))