import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import os
from scipy.stats import ortho_group
# os.environ["TORCH_HOME"]="./backbone_model"
torch.set_printoptions(threshold=np.inf)

class Attention_Module(nn.Module):
    def __init__(self):
        super(Attention_Module,self).__init__()

    def forward(self,q,k,v,ori,mode,gate=False):
        '''
        q = [B,Nq,outsize*outsize] / [clsnum,outsize*outsize]
        k,v = list([Nc,outsize*outsize]) / [B,Nc,outsize*outsize]
        ori : need to pad size / None
        mode = 'CE' or 'pattern'
        '''
        if mode == 'CE':
            bat = k.shape[0]
            q = F.normalize(q,dim=-1)
            k = F.normalize(k,dim=-1)
            q = q.unsqueeze(0).expand(bat,-1,-1)
            # alpha = F.softmax(torch.bmm(q,k.transpose(2,1)),-1) # [B,clsnum,Nc]
            alpha = torch.bmm(q,k.transpose(2,1)) # [B,clsnum,Nc]
            # if gate:
            #     thres = self.cal_thres(alpha.detach()) # [B,clsnum]
            #     alpha[alpha < thres.unsqueeze(-1)] = 0
            #     assert ((alpha!=0).sum(-1)>0).sum() == alpha.shape[0]*alpha.shape[1],print(alpha)
            # else:
            #     alpha[alpha < 0] = torch.sigmoid(10*alpha)
            final_v = (alpha.unsqueeze(-1) * v.unsqueeze(1)).sum(-2)
            return final_v,alpha

        elif mode == 'pattern':
            # d = q.shape[-1]
            alpha = []
            alpha_pad = []
            for i in range(q.shape[0]):
                # attw = F.softmax(torch.matmul(q[i],k[i].T) / np.sqrt(d),-1) # softmax
                attw = torch.matmul(F.normalize(q[i],dim=-1),F.normalize(k[i],dim=-1).T)
                alpha.append(attw) # [Nq,Nc]
                alpha_pad.append(F.pad(attw,(0,ori-attw.shape[-1],0,0),"constant",0))
            final_v = self.cal_value(alpha,v)
            alpha_pad = torch.stack(alpha_pad,0) # [B,Nq,ori]
            # mask = (alpha_pad == -1)[:,0] # [B,ori]
            return final_v,alpha_pad

        
    def cal_value(self,alpha,v):
        # zero_vec = torch.zeros([v[0].shape[-1]]).cuda() # [outsize*outsize,]
        fvalue = []
        for i in range(len(v)):
            fvalue.append(torch.matmul(alpha[i],v[i])) #[p_num,outsize*outsize]
        fvalue = torch.stack(fvalue,0) # [B,p_num,outsize*outsize]
        return fvalue
    
    def cal_thres(self,alpha):
        bat,clsnum = alpha.shape[0],alpha.shape[1]
        alpha = alpha.reshape(bat*clsnum,-1)
        allset = torch.sort(alpha,-1)[0]
        maxdis = torch.zeros(allset.shape[0]).cuda()
        thres = torch.zeros(allset.shape[0]).cuda() * -1
        for i in range(1,allset.shape[1]-1):
            lset = allset[:,:i]
            rset = allset[:,i:]
            dis = torch.pow(torch.mean(lset,-1) - torch.mean(rset,-1),2)
            ind = maxdis < dis
            maxdis[ind] = dis[ind]
            thres[ind] = allset[ind,i]
        thres = thres.reshape(bat,clsnum)
        return thres




class Proposed_Model(nn.Module):
    def __init__(self,outsize,clsnum,p_num,shrink,avgpool,choose_c,choose_value_mode,K):
        super(Proposed_Model, self).__init__()
        self.outsize = outsize
        self.possize = outsize * outsize
        self.clsnum = clsnum
        self.p_num = p_num
        self.shrink = shrink
        self.choose_c =choose_c
        self.choose_v = choose_value_mode
        self.K = K
        self.att_module = Attention_Module()
        self.enc = torchvision.models.vgg19(pretrained=True)
        # self.enc = nn.Sequential(*list(self.enc.children())[:-2],nn.AdaptiveAvgPool2d(outsize)) # output(512,outsize,outsize)
        self.enc = nn.Sequential(*(list(list(self.enc.children())[0].children())[:-2]),nn.AdaptiveAvgPool2d(outsize))# output(512,outsize,outsize)
        if shrink:
            self.shrlayer = nn.Conv2d(512,128,1) # output(128,outsize,outsize)
            outdim = 128
        else:
            self.shrlayer = None
            outdim = 512
        self.querygen = nn.Sequential(
                            nn.Linear(outsize*outsize,256),
                            # nn.LayerNorm(256),
                            nn.ReLU(),
                            nn.Linear(256,256),
                            # nn.LayerNorm(256),
                            nn.ReLU(),
                            nn.Linear(256,p_num*outsize*outsize)
                        )
        
        self.querydisc = self.querydisc = nn.Sequential(
                            nn.Linear((1+p_num)*outsize*outsize,256),
                            # nn.LayerNorm(256),
                            nn.ReLU(),
                            nn.Linear(256,256),
                            # nn.LayerNorm(256),
                            nn.ReLU(),
                            nn.Linear(256,1),
                            # nn.Sigmoid()
                        )
        self.feedlayer = nn.Sequential(
            nn.LayerNorm(outsize*outsize),
            # nn.Linear(outsize*outsize+self.possize,outsize*outsize),
            nn.Linear(outsize*outsize,outsize*outsize),
            nn.GELU(),
            nn.Linear(outsize*outsize,outsize*outsize)
        )
        # self.register_buffer('queue',torch.randn(outdim,self.K))

        # self.cls_token = nn.Parameter(torch.randn(self.clsnum,outsize*outsize))
        self.register_buffer('cls_token',torch.from_numpy(ortho_group.rvs(outsize*outsize)[:self.clsnum]).float())
        self.pos_emb = nn.Parameter(torch.randn(1,outdim,self.possize))

        self.register_buffer('queue',torch.randn(outsize*outsize,self.K))
        self.register_buffer('queuelabel',torch.ones(self.K)*-1)
        self.queue = F.normalize(self.queue,dim=0)
        self.register_buffer('queue_ptr',torch.zeros(1,dtype=torch.long))
        # self.register_buffer('ce_queue',torch.randn(outdim,self.K))
        # self.ce_queue = F.normalize(self.ce_queue,dim=0)
        # self.register_buffer('ce_queue_ptr',torch.zeros(1,dtype=torch.long))
        
    def forward(self,x,label,gate,mode):       
        
        fea_map = self.generate_feamap(x)
        fea_flatten = fea_map.flatten(start_dim=2) # [B,512/128,outsize*outsize]
        bat,ori_Nc,_ = fea_flatten.shape
        fea_add_pos = self.feedlayer((fea_flatten + self.pos_emb).reshape(bat*ori_Nc,-1)).reshape(bat,ori_Nc,-1)
        if self.choose_c and gate:
            cls_v,cls_alpha = self.att_module(self.cls_token,fea_add_pos,fea_add_pos,None,'CE',True)
        else:
            cls_v,cls_alpha = self.att_module(self.cls_token,fea_add_pos,fea_add_pos,None,'CE',False)
        
        if mode == 'init':
            # cls_pred = cls_v.norm(p=2,dim=-1,keepdim=False)
            cls_pred = cls_v.mean(-1,keepdim=False)
            # ind = torch.argmax(cls_v.detach().mean(-1),-1)
            max_v = cls_v[torch.arange(len(cls_v)).tolist(),label.view(-1).tolist()] #[B,out*out]
            return cls_pred,cls_alpha,max_v

        elif mode == 'all':
            # cls_pred = cls_v.norm(p=2,dim=-1,keepdim=False)
            cls_pred = cls_v.mean(-1,keepdim=False)
            max_v = cls_v[torch.arange(len(cls_v)).tolist(),label.view(-1).tolist()] #[B,out*out]
            all_token = self.get_clstoken(None)
            # token = all_token.index_select(0,label.view(-1))
            all_q = self.querygen(all_token)
            query = all_q.index_select(0,label.view(-1)).reshape(-1,self.p_num,self.outsize*self.outsize)
            full_q_pred = self.cal_discri_pred(all_token,all_q,label)

            choose_v,alpha,final_v,neg_query = self.cal_data_pattern(cls_alpha,query,label,fea_add_pos)
            return cls_pred,full_q_pred,choose_v,final_v,neg_query,cls_alpha,alpha,max_v

        elif mode == 'infer':
            all_token = self.get_clstoken(None)
            all_q = self.querygen(all_token)
            query = all_q.index_select(0,label.view(-1)).reshape(-1,self.p_num,self.outsize*self.outsize)
            choose_v,alpha,final_v,neg_query = self.cal_data_pattern(cls_alpha,query,label,fea_add_pos)
            return choose_v,all_q,cls_v
    
    
    def cal_data_pattern(self,cls_alpha,query,label,fea_add_pos):
        
        if self.choose_c:
            att_kv_mod = self.choose_channel(label,cls_alpha,fea_add_pos)
        else:
            att_kv_mod = list(torch.unbind(fea_add_pos,0))
        ori_Nc = fea_add_pos.shape[1]
        final_v,alpha = self.att_module(query,att_kv_mod,att_kv_mod,ori_Nc,'pattern',True) # [B,p_num,outsize*outsize] [B,Nq,Nc]
        choose_v,neg_query,cho_alpha = self.choose_value(final_v,query,alpha,self.choose_v)
        return choose_v,cho_alpha,final_v,neg_query

    def cal_discri_pred(self,all_token,all_q,label):
        
        # all_w = all_w / all_w.abs().max(-1,keepdim=True)[0]
        # w = all_token.index_select(0,label.view(-1))

        pair = self.build_w_fullquery_pair(all_token,all_q)
        full_q_pred = self.cal_batch_fullpair_discri_score(pair,label)# [B,clsnum]
        
        # pair = self.build_w_singlequery_pair(w,all_q)
        # single_q_pred = self.cal_batch_singlepair_discri_score(pair) # [B,pnum,clsnum]
        
        return full_q_pred#,single_q_pred

    def generate_feamap(self,x):
        fea_map = self.enc(x) # ->[B,-1,outsize,outsize]
        if self.shrink:
            fea_map = self.shrlayer(fea_map)

        return fea_map

    def generate_query_according2label(self,label):
        token = self.get_clstoken(label)
        # w  = w / w.abs().max(-1,keepdim=True)[0]
        query = self.querygen(token) # ->[B,p_num*outsize*outsize]
        query = query.reshape(-1,self.p_num,self.outsize*self.outsize)

        return query


    def cal_batch_fullpair_discri_score(self,pair,label):
        pair_pred = self.querydisc(pair.reshape(self.clsnum*self.clsnum,-1)).reshape(self.clsnum,self.clsnum)
        q_pred = pair_pred.index_select(0,label.view(-1)) # ->[B,clsnum]

        return q_pred
        
    def cal_batch_singlepair_discri_score(self,pair):

        assert pair.shape[1] * pair.shape[2] ==  self.p_num * self.clsnum
        pair = pair.reshape(pair.shape[0]*pair.shape[1]*pair.shape[2],-1)
        q_pred = self.querydisc(pair).reshape(-1,self.p_num,self.clsnum) # ->[B,p_num,clsnum]
        return q_pred

    def build_w_fullquery_pair(self,all_w,all_q):
        
        mod_all_w = all_w.unsqueeze(1).expand(-1,self.clsnum,-1) # ->[clsnum,clsnum,-1]
        mod_all_q = all_q.unsqueeze(0).expand(self.clsnum,-1,-1) # ->[clsnum,clsnum,p_num*outsize*outsize]
        pair = torch.cat([mod_all_w,mod_all_q],-1)

        return pair
    
    def build_w_singlequery_pair(self,w,all_q):

        all_q = all_q.reshape(self.clsnum,self.p_num,-1)
        zero_vec = torch.zeros(w.shape[0],self.clsnum,all_q.shape[-1]*(self.p_num-1)).cuda() # ->[B,clsnum,outsize*outsize*(pnum-1)]
        w = w.unsqueeze(1).expand(-1,self.clsnum,-1) # [B,clsnum,-1]
        concat_list = []
        for p in range(all_q.shape[1]):
            cat = [w,all_q[:,p].unsqueeze(0).expand(w.shape[0],-1,-1)]
            cat.insert(1,zero_vec[:,:,:p])
            cat.insert(-1,zero_vec[:,:,p:])
            concat_list.append(torch.cat(cat,-1))
        concat_list = torch.stack(concat_list,1) # [B,pnum,clsnum,-1]
        return concat_list


    def get_clstoken(self,label):
        '''
        get clstoken according to the class of data

        sw:[B,-1]
        '''
        token = self.cls_token.detach()
        if label is None:
            return token
        sw = token.index_select(0,label.view(-1))
        
        return sw

    def choose_channel(self,label,alpha,kv):
        alpha = alpha[torch.arange(alpha.shape[0]).tolist(),label.view(-1)] # [B,Nc]
        # print(((alpha!=0).sum(-1) > 0).sum(-1))
        alpha = (alpha * kv.mean(-1))>0
        # assert ((alpha!=0).sum(-1) > 0).sum() == len(alpha)
        
        fea_list = []
        for i in range(alpha.shape[0]):
            if alpha[i].sum() != 0:
                fea_list.append(kv[i][alpha[i]])
            else:
                fea_list.append(kv[i])

        return fea_list
    
    def choose_value(self,v,q,alpha,mode):
        '''
        v [B,p_num,outsize*outsize]
        q [B,p_num,outsize*outsize]
        alpha [B,p_num,Nc]
        '''
        assert mode in ['hard','soft'],"mode value must be 'soft' or 'hard' "
        # norm = v.norm(p=2,dim=-1,keepdim=False)
        norm = v.mean(-1,keepdim=False)
        if mode == 'hard':
            indlist = norm.argmax(-1).tolist()
            cho_alpha = alpha[torch.arange(alpha.shape[0]).tolist(),indlist] # [B,oriNc]
            v = v[torch.arange(v.shape[0]).tolist(),indlist] # [B,outsize*outsize]
            neg_q = torch.stack([torch.stack([q[j,i] for i in range(self.p_num) if i != indlist[j]],0) for j in range(v.shape[0])],0) #[B,p_num-1,outsize*outsize]
        if mode == 'soft':
            v = torch.bmm(F.softmax(norm,-1).unsqueeze(0),v).squeeze(1) # [B,outsize*outsize]
            neg_q = None
            cho_alpha = None
            
        return v,neg_q,cho_alpha

    def build_query_label(self,label):
        '''
        q_label : [B,clsnum]
        '''
        eye = torch.eye(self.clsnum).long()
        q_label = eye.index_select(0,label.view(-1))

        return q_label

    def dequeue_and_enqueue(self, value, label, mode='all'):
        '''
        value : [B,outsize*outsize] unnormalized
        '''

        batch_size = value.shape[0]
        value = F.normalize(value,dim=-1)
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0,print(self.K,batch_size)
        # replace the keys at ptr (dequeue and enqueue)
        if mode == 'all':
            ptr = int(self.queue_ptr)
            self.queue[:, ptr:ptr + batch_size] = value.T
            self.queuelabel[ptr:ptr + batch_size] = label
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.queue_ptr[0] = ptr
        elif mode == 'init':
            ptr = int(self.ce_queue_ptr)
            self.ce_queue[:, ptr:ptr + batch_size] = value.T
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.queue_ptr[0] = ptr


        
