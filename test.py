import torch
from torch.utils.data.dataloader import DataLoader
from model import Proposed_Model
import os
from data import Mod_Image_Floder,DA
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from utils import DataParallelPassthrough
np.set_printoptions(threshold=np.inf)

exist_dataset = ['CIFAR100','ILSVRC2012','Place365','Travel20','ILSVRC2012_smallSet']
mean_dict = {
    'CIFAR100':[0.5071, 0.4867, 0.4408],
    'ILSVRC2012':[0.485, 0.456, 0.406],
    'Place365':[0.485, 0.456, 0.406],
    'Travel20':[0.485, 0.456, 0.406],
    'ILSVRC2012_smallSet':[0.485, 0.456, 0.406]
}
std_dict = {
    'CIFAR100':[0.2675, 0.2565, 0.2761],
    'ILSVRC2012':[0.229, 0.224, 0.225],
    'Place365':[0.229, 0.224, 0.225],
    'Travel20':[0.229, 0.224, 0.225],
    'ILSVRC2012_smallSet':[0.229, 0.224, 0.225]
}



class Test():
    def __init__(self,):
        self.batchsize = 512
        self.outsize = 14
        self.clsnum = 20
        self.p_num = 5
        self.shrink = False
        self.avgpool = True
        self.choose_c = True
        self.choose_value_mode = 'hard'
        self.K = 4096
        self.net = Proposed_Model(self.outsize,self.clsnum,self.p_num,
                                self.shrink,self.avgpool,self.choose_c,self.choose_value_mode,self.K)
        self.net = self.net.cuda()
        self.savedir = './save'
        self.data = 'CIFAR100'
        self.insize=448
        # if self.data in exist_dataset:
        self.mean = mean_dict[self.data]
        self.std = std_dict[self.data]
        self.trans = DA(self.mean,self.std,self.insize,'simple')
        rdir = os.path.join('./data',self.data,'test')
        self.traindata = Mod_Image_Floder(rdir,self.insize,self.mean,self.std,transform=self.trans)
        self.DL = DataLoader(self.traindata,self.batchsize,'False',drop_last=False,num_workers=8)
        
    def run_the_net(self,x, label):
        all_v = None
        choose_v,all_q,all_v = self.net(x,label,None,'infer')
        return choose_v,all_q,all_v

    def load_model(self,epo=None):
        if epo == None:
            epo = 0
            # if don't give the specific epo, then load the last model by default
            filelist =  os.listdir(self.savedir)
            for file in filelist:
                if file.split('.')[-1] != 'pt':
                    continue
                epo = max(epo,int(file.split('.')[0].split('_')[1]))

        ckpt = torch.load(os.path.join(self.savedir,'model_'+str(epo)+'.pt'))
        self.net.load_state_dict(ckpt['model'])
        self.net = DataParallelPassthrough(self.net)
        # self.opt.load_state_dict(ckpt['opt'])
        # if ckpt['use_dp']:
        #     print('use data parallel strategy')
        #     self.dp()
    def test(self):
        with torch.no_grad():
            epo = 102
            self.load_model(epo)
            self.net.eval()
            pattern_value = []
            label_list = []
            cls_value = []
            logit = []
            # pattern_fea = [[] for _ in range(self.clsnum)]
            for data,_,label in self.DL:
                data = data.cuda()
                label = label.cuda()
                data = data[label < 5]
                label = label[label < 5]
                if len(data) == 0:
                    continue
                p_v,p_fea,all_v = self.run_the_net(data,label)
                all_v = all_v.cpu()
                ind = torch.argmax(all_v.mean(-1),-1)
                # print(all_v.mean(-1))
                # print(ind)
                all_v = all_v[torch.arange(len(all_v)).tolist(),label.view(-1).tolist()]
                p_v = F.normalize(p_v,dim=-1)
                all_v = F.normalize(all_v,dim=-1).numpy()
                p_v = p_v.cpu().numpy()
                # p_fea = p_fea.cpu().numpy()
                # all_v = all_v.cpu().numpy()
                # if len(pattern_fea[label]) == 0:
                    # pattern_fea[label].append(p_fea)
                pattern_value.append(p_v)
                cls_value.append(all_v)
                label_list.append(label.cpu().numpy())
                logit.append(ind.numpy())
        pattern_value = np.concatenate(pattern_value,0)
        label_list = np.concatenate(label_list,0)
        cls_value = np.concatenate(cls_value,0)
        logit = np.concatenate(logit,0)
        # # pattern_fea = pattern_fea
        return label_list,pattern_value,cls_value,logit
    def get_all_q(self,):
        with torch.no_grad():
            all_token = self.net.get_clstoken(None)
            all_q = F.normalize(self.net.querygen(all_token),dim=-1).cpu().numpy()
            all_token = F.normalize(all_token,dim=-1).cpu().numpy()
            all_q = all_q.reshape(-1,self.p_num,self.outsize*self.outsize) #[clsnum,p_num,out*out]
            # all_q = F.normalize(all_q,dim=-1)
            # all_token = F.normalize(all_token,dim=-1).cpu().numpy()
        return all_q,all_token


T = Test()
label_list,pattern_value,cls_value,logit = T.test()
print(sum(logit == label_list) / len(logit))
query,clstoken = T.get_all_q()

clstoken = clstoken
now_clsnum = len(clstoken)
clsavg = []
# for l in range(5):
#     clsavg.append(np.mean(pattern_value[label_list == l],0,keepdims=False))
# clsavg = np.stack(clsavg,0)

# clsnum,pnum,_ = query.shape
# query = query.reshape(-1,query.shape[-1])
# all_value = np.concatenate([pattern_value,query],0)
# dis_mat = np.clip(1. - (all_value.dot(all_value.T))**2,0,None)
# all_value = np.concatenate([cls_value,clstoken],0)
dis_mat = np.clip(1. - (pattern_value.dot(pattern_value.T)),0,None)
# print(dis_mat)
# tsne = TSNE(n_iter=15000, metric="precomputed",perplexity=40)
# res = tsne.fit_transform(dis_mat)
# pattern_res = res[:len(label_list)]
# clsres = res[len(label_list):len(label_list)+now_clsnum]
# query_res = res[len(label_list)+now_clsnum:].reshape(clsnum,pnum,-1)
# avgres = res[len(label_list)+5:]

# pattern_res = res[:len(label_list)]
# query_res = res[len(label_list):].reshape(clsnum,pnum,-1)
# dis_mat = np.clip(1. - (cls_value.dot(cls_value.T))**2,0,None)
tsne = TSNE(n_iter=15000, metric="precomputed",perplexity=50)
res = tsne.fit_transform(dis_mat)
# value = res[:len(cls_value)]
# token = res[len(cls_value):]
plt.figure()
plt.scatter(res[:,0],res[:,1],s=5,c=label_list)
# plt.scatter(token[:,0],token[:,1],s=10,c='red')
plt.savefig('./allp.png')
plt.close()
print('all wc/pc complete')
# for l in range(now_clsnum):
    # now_p_v = pattern_value[label_list == l]
    # now_query = query[l]
    # all_value = np.concatenate([now_p_v,now_query],0)
    # dis_mat = np.clip(1. - (all_value.dot(all_value.T))**2,0,None)
    # tsne = TSNE(n_iter=15000, metric="precomputed",perplexity=30)
    # res = tsne.fit_transform(dis_mat)
    # p_v_res = res[:len(now_p_v)]
    # query_res = res[len(now_p_v):]
    # plt.figure()
    # plt.scatter(p_v_res[:,0],p_v_res[:,1],s=10,c=[l]*len(p_v_res))
    # plt.scatter(query_res[:,0],query_res[:,1],s=15,c='red')
    # plt.savefig('./{}.png'.format(str(l)))
    # plt.close()

    # plt.figure()
    # xmin = np.min(res[label_list == l,0]) - 20
    # xmax = np.max(res[label_list == l,0]) + 20
    # ymin = np.min(res[label_list == l,1]) - 20
    # ymax = np.max(res[label_list == l,1]) + 20
    # plt.xlim(xmin,xmax)
    # plt.ylim(ymin,ymax)
    # plt.scatter(res[label_list==l,0],res[label_list==l,1],s=10,c=[l]*sum(label_list==l))
    # # plt.scatter(res[:,0],query_res[:,1],s=5,c='red')
    # plt.savefig('./{}.png'.format(str(l)),dpi=1500)
    # plt.close()
