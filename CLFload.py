is_green=True

# 여러 분류 모델의 구조가 사전에 정의되어 있는 코드입니다.

try:
    import torch
    import torch.nn as nn
    from config_load import config
    import numpy as np
    import random
    
    # Seed setting
    def set_seed(num=42):
        torch.manual_seed(num)
        torch.cuda.manual_seed(num)
        torch.cuda.manual_seed_all (num)
        np.random.seed(num)
        random.seed(num)

        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True

    set_seed(config['seed'])
    device=config['device']
    torch.set_default_device(device)
    clftype=config['CLFmodel']
    clfconf=config[clftype+'_config']
    is_debug=config['debug']
    
    class Linear_clf(nn.Module):
        def __init__(self, device=device, febase=config['FEmodel']):
            super(Linear_clf, self).__init__()
            self.modtype='Linear_clf'
            self.dim=768
            if config['FEmodel']=='wtv_kr' or config['FEmodel']=='wtv_base':
                self.dim=512
            self.l1=nn.Linear(self.dim, 1)
            self.to(device)
            print(' Linear clf loaded.')
        def forward(self, data):
            #print('input:', data)
            if is_debug:
                print('Linear clf forward pass working now.')
            batchsize=data.shape[0]
            data=torch.mean(data, dim=1)
            data=self.l1(data)
            #print('output:', data)
            return data
            
    class Conv_clf(nn.Module):
        def __init__(self, ):
            super(Conv_clf, self).__init__()
            self.modtype='Linear_clf'
            self.dim=768
            self.to(device)
            print(' Conv clf loaded.')
        pass
    class Attention_clf(nn.Module):
        def __init__(self, hdim, layer_num, token_number, nhead=32, febase=config['FEmodel']):
            super(Attention_clf, self).__init__()
            self.hdim=hdim
            self.dim=768
            self.token_number=token_number
            if config['FEmodel']=='wtv_kr' or config['FEmodel']=='wtv_base':
                self.dim=512
            self.modtype='Attention_clf'
            for _ in range(layer_num):#############################여기수정
                self.pdtoken=nn.Parameter(torch.randn( token_number, self.hdim, requires_grad=True))
            if hdim!=self.dim:
                self.lin1=nn.Linear(self.dim, hdim, device=device)
            self.layer=nn.ModuleList()
            self.lastlin=nn.Linear(self.hdim, 1)
            self.lastlin2=nn.Linear(token_number, 1)
            for i in range(layer_num):
                self.layer.append(nn.MultiheadAttention(embed_dim=self.hdim, num_heads=nhead))
                # if i-1==layer_num:
                #     break
                self.layer.append(nn.GELU())
            #self.layer.append(nn.Sigmoid())
            self.to(device)
            print(' Cross-attention clf loaded.')
            
        def forward(self, data):
            batchsize=data.shape[0]
            data=data.permute(1, 0, 2)
            logits=[]
            for _ in range(batchsize):
                logits.append(self.pdtoken)
            logits=torch.stack(logits, dim=1)
            #logits=self.pdtoken
            if is_debug:
                print('cross-attention clf forward pass')
            i=0
            
            if self.hdim!=self.dim:
                data=self.lin1(data)
            for layer in self.layer:
                #print('logits', logits.shape)
                if i%2==0:
                    #print('logits:',logits.shape, 'data', data.shape)
                    logits, attbin = layer(logits, data, data)
                    #logits.append(  layer(self.pdtoken[int(i/2)], data))
                else:
                    logits=layer(logits)
                i=i+1
            logits=self.lastlin(logits)
            #print('logits after ll1', logits.shape)
            logits=logits.squeeze(2)
            #print('logits after squeeze', logits.shape)
            logits=logits.permute(1, 0)
            #print('logits after perm', logits.shape)
            logits=self.lastlin2(logits)
            #print('logits after ll2', logits.shape)
            return logits
            
    class selfattclf(nn.Module):
        def __init__(self, d_model=256, layers=2, nhead=16, febase=config['FEmodel']):
            super(selfattclf, self).__init__()
            self.dim=768
            if config['FEmodel']=='wtv_kr':
                self.dim=1024
            self.l1=nn.Linear(self.dim, d_model)
            enclayer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)#, device='cuda')
            self.clf1=nn.TransformerEncoder(enclayer, layers, norm=None)#, device='cuda')
            self.activ=nn.GELU()
            self.lin=nn.Linear(d_model, 1)#, device='cuda')
            self.to(device)
            print(' Self-Attnetion clf loaded.')
        def forward(self, data):
            if is_debug:
                print('self-attention clf forward pass')
            data=self.l1(data) # edited
            data=self.activ(data)
            data=self.clf1(data)[:, 0, :]
            data=self.activ(data)
            data=self.lin(data)
            #print('clf output shape:', data.shape)
            #print('data.max:', data.max())
            return data

    class External_Attention():
        def __init__(self,):
            super(External_Attention, self).__init__()
            self.modtype='External_Attention'
        pass
    if clftype=='cross-attention':
        clfmodel=Attention_clf(**clfconf)
    if clftype=='linear':
        clfmodel=Linear_clf(**clfconf)
    if clftype=='self-attention':
        clfmodel=selfattclf(**clfconf)
    
except Exception as e:
    print('Exception', e, 'from CLFload.')
    is_green=False