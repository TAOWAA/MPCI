from urllib.parse import ParseResultBytes
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from PIL import ImageFilter
import argparse
import math


class cal_entropy_loss():
    '''
    standard CE loss
    '''
    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss()
    def __call__(self,x,y):
        '''
        x : [B,C]
        y : [B,] unnormalized
        '''
        return self.loss(x,y.view(-1))
        

class cal_const_loss():
    '''
    CE constant loss
    '''
    def __call__(self,pred1,pred2):
        '''
        pred1,pred2 : [B,C]
        pred2 guide pred1
        '''
        
        return F.kl_div(F.log_softmax(pred1,dim=-1),F.softmax(pred2,dim=-1),reduction='batchmean')

class cal_const2_loss():
    '''
    attention CE constant loss
    '''
    def __call__(self,alpha1,alpha2,label):
        '''
        alpha1,alpha2 : [B,clsnum,C]
        alpha2 guide alpha1
        '''
        bat,_,_ = alpha1.shape
        alpha1 = alpha1[torch.arange(bat).tolist(),label.view(-1)]
        alpha2 = alpha2[torch.arange(bat).tolist(),label.view(-1)]
        # alpha1 = alpha1.reshape(bat*clsnum,-1)
        # alpha2 = alpha2.reshape(bat*clsnum,-1)

        return F.kl_div(F.log_softmax(alpha1,dim=-1),F.softmax(alpha2,dim=-1),reduction='batchmean')
        
class cal_att_const_loss():
    '''
    attention module constant loss
    '''
    def __call__(self,alpha1,alpha2):
        '''
        alpha1,alpha2 : list([Nq,Nc])
        alpha2 guide alpha1
        '''
        # loss = 0
        # bat = len(alpha1)
        return F.kl_div(torch.log(alpha1 + 1e-10),alpha2+1e-10,reduction='batchmean')
        # for i in range(bat):
        #     loss += F.kl_div(alpha1,alpha2,reduction='batchmean')
        # loss /= bat
        # return loss

class cal_mod_contra_loss():
    '''
    modified contrastive loss
    '''
    
    def __call__(self,o_v,t_v,queue,neg_q):
        '''
        o_v,t_v : [B,outsize*outsize]
        queue : [outsize*outsize,K]
        '''
        b = o_v.shape[0]
        o_v = F.normalize(o_v,dim=-1)
        t_v = F.normalize(t_v,dim=-1)
        # neg_sim = torch.pow(torch.matmul(o_v,queue),2)

        # pos_dis = torch.clamp(1 - torch.pow((o_v * t_v).sum(-1),2),min=0.,max=1.) #[B,]
        pos_dis = torch.clamp(1 - (o_v * t_v).sum(-1),min=0.) #[B,]

        neg_dis = []
        # for l in range(b):
        #     neg_sim.append(torch.pow(torch.matmul(o_v[l].unsqueeze(0),queue[l]),2).mean().detach())
        # neg_sim = torch.Tensor(neg_sim)
        for l in range(o_v.shape[0]):
            # neg_sim = torch.pow(torch.matmul(o_v[l].unsqueeze(0),queue[l]),2)
            neg_sim = torch.matmul(o_v[l].unsqueeze(0),queue[l])
            neg_sim = torch.clamp(1 - neg_sim,min=0.).mean()
            neg_dis.append(neg_sim)
        neg_dis = torch.stack(neg_dis,0)
        # print(pos_dis)
        # print(neg_dis)
        # print(neg_dis.device)
        if neg_q is not None:
            # neg_q_sim = torch.pow(torch.bmm(o_v.unsqueeze(1),neg_q.transpose(2,1)).squeeze(1),2)*0.9 # [B,pnum-1]
            neg_q_sim = torch.bmm(o_v.unsqueeze(1),neg_q.transpose(2,1)).squeeze(1)*0.9 # [B,pnum-1]
            neg_dis = torch.cat([neg_sim,torch.clamp(1-neg_q_sim,min=0.)],-1)

        # neg_dis = (1 - neg_sim) * 0.7
        # sim_mat = torch.clamp(torch.pow(torch.matmul(o_v,t_v.T),2),min=0.,max=1.) # [B,B]
        # pos_dis = 1 - sim_mat.gather(-1,torch.arange(b).long().view(-1,1).cuda()) # [B,1]
        # margin = (neg_dis + pos_dis.mean().item()).unsqueeze(1).cuda() # [B,1]
        # neg_ind = torch.Tensor([[i for i in range(b) if i != j ] for j in range(b)]).long().cuda()
        # neg_dis = 1 - sim_mat.gather(-1,neg_ind) # [B,B-1]

        margin = (neg_dis + pos_dis).unsqueeze(1).detach() # [B,1]
        pos_loss = pos_dis**2/(2*b) 
        neg_loss = ((1/margin) * torch.pow(torch.clamp(torch.pow(neg_dis,0.5)*(margin-neg_dis),min=0.),2)).sum(-1) / (2*b)
        loss = (pos_loss + neg_loss).mean()
        return loss


class cal_std_contra_loss():
    '''
    standard contrastive loss
    '''
    def __init__(self,tau):
        self.tau = tau
        self.loss = nn.CrossEntropyLoss()

    def __call__(self,o_v,t_v,queue,neg_q):
        '''
        o_v,t_v : [B,outsize*outsize]
        queue : [outsize*outsize,K]
        neg_q : [B,pnum-1,outsize*outsize]
        '''
        o_v = F.normalize(o_v,dim=-1)
        t_v = F.normalize(t_v,dim=-1)
        # pos_sim = torch.pow((o_v * t_v).sum(-1,keepdim=True),2) # [B,1]
        pos_sim = (o_v * t_v).sum(-1,keepdim=True) # [B,1]
        # neg_sim = torch.pow(torch.matmul(o_v,queue),2) # [B,K]
        loss = 0
        for l in range(o_v.shape[0]):
            # neg_sim = torch.pow(torch.matmul(o_v[l].unsqueeze(0),queue[l]),2)
            neg_sim = torch.matmul(o_v[l].unsqueeze(0),queue[l])
            label = torch.zeros(1).cuda()
            all_sim = torch.cat([pos_sim[l].unsqueeze(0),neg_sim],-1) / self.tau
            loss += self.loss(all_sim,label.long())
        loss = loss / o_v.shape[0]
        # neg_sim = torch.cat(neg_sim,0)
        
        if neg_q is not None:
            # neg_q_sim = torch.pow(torch.bmm(o_v.unsqueeze(1),neg_q.transpose(2,1)).squeeze(1),2)*0.5 # [B,pnum-1]
            neg_q_sim = torch.bmm(o_v.unsqueeze(1),neg_q.transpose(2,1)).squeeze(1) * 0.5 # [B,pnum-1]
            neg_sim = torch.cat([neg_sim,neg_q_sim],-1)
        
        # label = torch.zeros([neg_sim.shape[0]]).cuda()
        # all_sim = torch.cat([pos_sim,neg_sim],-1)/self.tau
        # loss = self.loss(all_sim,label.long())
        
        return loss


class cal_ce_constra_loss():
    def __init__(self,tau):
        self.tau = tau
        self.loss = nn.CrossEntropyLoss()
    def __call__(self,v,clstoken,label):
        '''
        v : [B,outsize*outsize]
        clstoken : [clsnum,outsize*outsize]
        label : [B,1]
        '''

        v = F.normalize(v,dim=-1)
        clstoken = F.normalize(clstoken,dim=-1)
        all_sim = torch.pow(torch.matmul(v,clstoken.T),2) / self.tau # [B,clsnum]
        
        label = label.view(-1)
        loss = self.loss(all_sim,label.long())
        return loss


class cal_MI_loss():
    '''
    mutual information loss
    '''
    def __init__(self):
        
        self.MIloss = nn.BCEWithLogitsLoss()
        
    def __call__(self,qpred,label):
        '''
        qpred : [B,clsnum]/[B,p_num,clsnum] undoing sigmoid 
        label : [B,]
        '''
        label = label.view(-1)
        neg_ind = torch.LongTensor(
            [[i for i in range(qpred.shape[-1]) if i != label[j]] for j in range(qpred.shape[0])]).cuda()
        
        if len(qpred.shape) == 2:
            pos = qpred.gather(-1,label.view(-1,1))
            neg = qpred.gather(1,neg_ind)
            pos_label = torch.ones_like(pos)
            neg_label = torch.zeros_like(neg)
            pos_loss = self.MIloss(pos,pos_label)
            neg_loss = self.MIloss(neg,neg_label)
        else:
            modlabel = label.view(-1,1,1).expand(-1,qpred.shape[1],-1)
            pos = qpred.gather(-1,modlabel).reshape(-1,1)
            neg = qpred.gather(-1,neg_ind.unsqueeze(1).expand(-1,qpred.shape[1],-1)).reshape(-1,1)
            pos_label = torch.ones_like(pos)
            neg_label = torch.zeros_like(neg)
            pos_loss = self.MIloss(pos,pos_label)
            neg_loss = self.MIloss(neg,neg_label)
            
        loss = (pos_loss + neg_loss) / 2

        return loss


class cal_dcp_loss():
    '''
    decouple loss
    '''
    def __init__(self,):
        pass
    def __call__(self,fq_pred,sq_pred,label):
        '''
        fq_pred : [B,clsnum]
        sq_pred : [B,pnum,clsnum]
        label : [B,]
        '''
        label = label.view(-1)
        loss = ((fq_pred.gather(-1,label.view(-1,1)) - sq_pred.sum(1).gather(-1,label.view(-1,1)))**2).mean()
        return loss
        

class cal_dcp2_loss():
    
    def __call__(self,v):
        '''
        v [B,p_num,outsize*outsize]
        '''
        # norm = v.norm(p=2,dim=-1,keepdim=False)
        norm = v.mean(-1,keepdim=False)
        
        P1 = F.softmax(norm,dim=-1) # [B,pnum]
        P2 = F.log_softmax(norm,dim=-1)
        loss = -(P1 * P2).sum(-1).mean()
        return loss

        

class cal_tri_loss():
    def __call__(self,alpha):
        '''
        alpha : [B,OriNc]
        '''
        # bat = len(alpha)
        # loss = 0
        loss = 0
        bat = alpha.shape[0]
        for b in range(bat):
            per_alpha = alpha[b][alpha[b] != 0]
            per_alpha = torch.pow(per_alpha,2)
            loss += (-1 * per_alpha * torch.log(per_alpha + 1e-10)).sum(-1)
        loss = loss / bat
        # for i in range(bat):
        #     loss += (-1 * alpha[i] * torch.log(alpha[i] + 1e-10)).sum(-1).mean()
        # loss /= bat
        return loss

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x




def str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def resume_train_model(args,T,stage,epo):
    if stage == 'init':
        T.train(epo,args.init_epo,'init')
        T.train(0,args.all_epo,'all')
    else:
        T.train(epo,args.all_epo,'all')


def train_model_from_start(args,T):
    if args.dp:
        T.dp()
    T.train(0,args.init_epo,'init')
    T.train(0,args.all_epo,'all')


class DataParallelPassthrough(nn.DataParallel):
    '''code from https://github.com/pytorch/pytorch/issues/16885'''
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class CustomDataParallel(nn.Module):
    def __init__(self, model, device):
        super(CustomDataParallel, self).__init__()
        self.model = nn.DataParallel(model, device)

    def forward(self, *input):
        return self.model(*input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)


def adjust_learning_rate(opt,epo,all_epo,init_lr,all_lr,mode):
    init_lr *= 0.5 * (1. + math.cos(math.pi * epo / all_epo))
    all_lr *= 0.5 * (1. + math.cos(math.pi * epo / all_epo))
    if mode == 'init':
        opt.param_groups[0]['lr'] = init_lr
    elif mode == 'all':
        opt.param_groups[0]['lr'] = init_lr
        opt.param_groups[1]['lr'] = all_lr


def sparse2coarse(class_to_index):
    '''
    code from https://github.com/ryanchankh/cifar100coarse
    '''
    
    # coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
    #                            3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
    #                            6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
    #                            0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
    #                            5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
    #                            16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
    #                            10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
    #                            2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
    #                           16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
    #                           18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                ['bottle', 'bowl', 'can', 'cup', 'plate'],
                ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                ['crab', 'lobster', 'snail', 'spider', 'worm'],
                ['baby', 'boy', 'girl', 'man', 'woman'],
                ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
    for i in range(len(classes)):
        for c in classes[i]:
            class_to_index[c] = i
        
    return class_to_index

