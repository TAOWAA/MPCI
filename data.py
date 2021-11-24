import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder 
from utils import GaussianBlur
from PIL import Image
import os
import numpy as np

class DA():
    def __init__(self,mean,std,in_size,mode):
        '''
        mode: 'simple' or 'FT'
        '''
        self.mean = mean
        self.std = std
        self.input_size = in_size
        normalize = transforms.Normalize(mean = self.mean,std = self.std)
        
        if mode == 'simple':
            self.trans = transforms.Compose([
                transforms.Resize([self.input_size,self.input_size]),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            #     transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            # ], p=0.8),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                normalize
            ])

        elif mode == 'FT':
            pass
        
        else:
            raise ValueError("'mode' value must be 'simple' or 'FT' ")

    def __call__(self,x):
        return self.trans(x)


class Mod_Image_Floder(ImageFolder):
    def __init__(self,root,insize,mean,std,transform):
        super().__init__(root,transform)
        self.basictrans = transforms.Compose([
            transforms.Resize([insize,insize]),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean,std = std)
            ])
        if 'CIFAR100' in root.split('/'):
            self.coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                                            3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                            6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                                            0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                                            5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                                            16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                                            10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                                            2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                                            16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                                            18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
            # self.fine_labels = [[] for i in range(20)]
            # for i in range(len(self.coarse_labels)):
            #     self.fine_labels[self.coarse_labels[i]].append(i)
        else:
            self.coarse_labels = None
        # print(self.coarse_labels)
        # print(self.fine_labels)

    def __getitem__(self, index):
        path,target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            t_img = self.transform(img)
            img = self.basictrans(img)
        else:
            img = self.basictrans(img)
            t_img = img
        if self.coarse_labels is not None:
            target = self.coarse_labels[target]
            # ctarget = self.coarse_labels[target]
            # ftarget = self.fine_labels[ctarget].index(target)
            # target = torch.Tensor(np.array([ftarget,ctarget])).long()
        return img,t_img,target



class OwnDataset(Dataset):
    def __init__(self,dir,datadir,labeldir,insize,trans,mean,std):
        super(OwnDataset, self).__init__()
        self.dir = dir
        self.datadir = datadir
        self.labeldir = labeldir
        self.mean = mean
        self.std = std
        if trans:
            self.trans = trans
        else:
            self.trans = None
        self.basictrans = transforms.Compose([
            transforms.Resize([insize,insize]),
            transforms.ToTensor(),
            transforms.Normalize(self.mean,self.std)
            ])

        self.datafile = os.listdir(os.path.join(dir,datadir))
        self.labelfile = os.listdir(os.path.join(dir,labeldir))

    def __getitem__(self, index):

        data = self.datafile[index]
        data = Image.open(os.path.join(self.dir,self.datadir,data))
        if self.trans is not None:
            t_data = self.trans(data)
            data = self.basictrans(data)
        else:
            data = self.basictrans(data)
            t_data = data
        label = self.labelfile[index]
        
        return data,t_data,label

    def __len__(self):
        
        return len(self.datafile)