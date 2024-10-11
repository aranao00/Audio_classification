import torch
import torch.nn as nn
import FEload
import CLFload
import dataloader
from FEload import femod
from CLFload import clfmodel as clfmod
from dataloader import Voice_loader as dlpd
import numpy as np
import random
from config_load import config
from printf import prif

# 모델 학습용 코드입니다.

datapath=''


# Seed setting
def set_seed(num=42):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all (num)
    np.random.seed(num)
    random.seed(num)

    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

# Load Config Data
device=config['device']
torch.set_default_device(device)
tasknum=config['task_number']
set_seed(config['seed'])

fetype=config['FEmodel']
modtype=config['CLFmodel']
epochs=config['epochs']
lr=config['lr']
batchsize=config['batchsize']
iters_per_ep=config['iter_per_epochs']



# Debugging
if FEload.is_green & CLFload.is_green & dataloader.is_green:
    print('Library all green.')
else:
    print(
'''#####################################
#  Check the error message please.  #
#####################################''')



# Pring Config
print(f'''
epochs              : {epochs}
learning rate       : {lr}
batchsize           : {batchsize}
preprocessing       : Wav2Vec 2.0 preprocessor

feature extractor   : {fetype}
classifier          : {modtype}

loss function       : BCE loss
lossadv function    : Cross Entropy Loss
classifier optimizer: Adam
''')
prif(f'''
epochs              : {epochs}
learning rate       : {lr}
batchsize           : {batchsize}
preprocessing       : Wav2Vec 2.0 preprocessor

feature extractor   : {fetype}
classifier          : {modtype}

loss function       : BCE loss
lossadv function    : Cross Entropy Loss
classifier optimizer: AdamW
''')

prif(config)
prif('\n######################################\n')


# Prepare Training
femod.eval()
lossfn=nn.BCEWithLogitsLoss()#weight=[0.5,0.5])
lossfn_adv=nn.CrossEntropyLoss()
optim=torch.optim.AdamW(clfmod.parameters(), lr=lr)
sigmoid=nn.Sigmoid()


# Start Training
for ep in range(epochs):
    
    
    # Training Loop
    clfmod.train()
    
    # Loop iters_per_ep times
    for iters in range(iters_per_ep):
        
        # Load data
        audio, label, metadata=dlpd.call_train(batchsize)
        audio=audio.to(device)
        label=label.to(device)
        
        with torch.no_grad():
        # Extract features
            if fetype=='hub_base' or fetype=='hub_kr':
                feature=femod(audio).last_hidden_state
            elif fetype=='wtv_base' or fetype=='wtv_kr':
                feature=femod(audio).extract_features
        
        # Forward
        score=clfmod(feature)
        loss=lossfn(score, label.unsqueeze(1))
        #loss_adv=lossfn_adv(advscore, metadata)
        loss_total=loss#+loss_adv
        #print(score)
        
        # Evaluate
        acc=0
        acc_pd=0
        acc_hc=0
        pred_pd=0
        pred_hc=0
        score=sigmoid(score)
        for i in range(batchsize):
            if i%2==0:
                if score[i]>=0.5:
                    acc+=1
                    acc_pd+=1
                    pred_pd+=1
                else:
                    pred_hc+=1
            else:
                if score[i]<=0.5:
                    acc+=1
                    acc_hc+=1
                    pred_hc+=1
                else:
                    pred_pd+=1
            i+=1
        acc= acc/batchsize*100
        if pred_pd==0:
            precision_pd=0
        else:
            precision_pd=acc_pd/pred_pd
        recall_pd=acc_pd/batchsize*2
        if pred_hc==0:
            precision_hc=0
        else:
            precision_hc=acc_hc/pred_hc
        recall_hc=acc_hc/batchsize*2
        if (precision_pd==0) & (recall_pd==0):
            fscr1=0
        else:
            fscr1= 2*(precision_pd*recall_pd)/(precision_pd+recall_pd)
        acc_pd=acc_pd/batchsize*200
        acc_hc=acc_hc/batchsize*200
        prif(ep, 'iter [', iters, '/', iters_per_ep, ']', loss_total.detach().item(), 'Train acc:', acc, 'pd', acc_pd, 'hc', acc_hc, 'f1', fscr1, 'P&R', precision_pd, ',', recall_pd)
        print (ep, 'iter [', iters, '/', iters_per_ep, ']', loss_total.detach().item(), 'Train acc:', acc, 'pd', acc_pd, 'hc', acc_hc, 'f1', fscr1, 'P&R', precision_pd, ',', recall_pd)
        
        
        # Optimize
        optim.zero_grad()
        loss_total.backward()
        #torch.nn.utils.clip_grad_norm_(clfmod.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()
        
        del audio, label
    #dlpd.shuffle()
    
    
    
    print(f'''######################################
          {ep:03d} : Start Validation
######################################''')
    
    
    # Validation
    with torch.no_grad():
        # Forward
        loss_ep=torch.zeros(1)
        clfmod.eval()
        audio,label,metadata = dlpd.call_test()
        
        if fetype=='hub_base' or fetype=='hub_kr':
            feature=femod(audio).last_hidden_state
        elif fetype=='wtv_base' or fetype=='wtv_kr':
            feature=femod(audio).extract_features
            
        score=sigmoid(clfmod(feature))
        #loss=lossfn(label, score)
        #loss_adv=lossfn_adv(advscore, metadata)
        #loss_total=loss#+loss_adv
        #loss_ep+=loss_total/audio.shape[0]
        
        
        torch.save(clfmod, f'{datapath}/{modtype}-{fetype}-task{tasknum}-{ep+1}ep.pth')
        prif( f'{datapath}/{modtype}-{fetype}-task{tasknum}-{ep+1}ep.pth saved.')
        
        acc=0
        acc_pd=0
        acc_hc=0
        pred_pd=0
        pred_hc=0
        for i in range(60):
            if i%2==0:
                if score[i]>=0.5:
                    acc+=1
                    acc_pd+=1
                    pred_pd+=1
                else:
                    pred_hc+=1
            else:
                if score[i]<=0.5:
                    acc+=1
                    acc_hc+=1
                    pred_hc+=1
                else:
                    pred_pd+=1
            i+=1
        if pred_pd==0:
            precision_pd=0
        else:
            precision_pd=acc_pd/pred_pd
        if pred_hc==0:
            precision_hc=0
        else:
            precision_hc=acc_hc/pred_hc
        recall_pd=acc_pd/30
        recall_hc=acc_hc/30
        if (precision_pd==0) & (recall_pd==0):
            fscr1=0
        else:
            fscr1=2*(precision_pd*recall_pd)/(precision_pd+recall_pd)
        #print(f'ep {ep+1}/{epochs} : {loss_ep.item:.8f}')
        prif('Accuracy', acc/60*100, acc, 'pd', acc_pd/30*100, 'hc', acc_hc/30*100, 'f1', fscr1, 'P&R', precision_pd, ',', recall_pd)
        print('Accuracy', acc/60*100, acc, 'pd', acc_pd/30*100, 'hc', acc_hc/30*100, 'f1', fscr1, 'P&R', precision_pd, ',', recall_pd)

print('done.')
