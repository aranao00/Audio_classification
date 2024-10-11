is_green=True
#try:

# 여러 사전 학습 모델을 로드하기 위한 코드입니다.

print('FE load : import libraries')
import torch
import torch.nn as nn
# Load model directly
from transformers import pipeline
from transformers import AutoProcessor, AutoModel
from config_load import config
import random
import numpy as np
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
print('FE load : loading processor')

cache_dir_hub='/tf/nasw/hubert'
cache_dir_wv2='/tf'#/wTvT'
cache_dir_hub_ls960='/tf'#/hubert'
cd='/tf'
modtype=config['FEmodel']

if modtype=='hub_kr':
    print('FE load : loading hubert-base-korean')
    femod = AutoModel.from_pretrained("team-lucid/hubert-base-korean", trust_remote_code=True, cache_dir=cd, local_files_only=True, device_map=device)

if modtype=='wtv_base':
    print('FE load : loading wav2vec2-base')
    femod = AutoModel.from_pretrained("facebook/wav2vec2-base-960h", trust_remote_code=True, cache_dir=cache_dir_wv2, local_files_only=True, device_map=device)
    
if modtype=='hub_base':
    print('FE load : loading hubert-base')
    femod = AutoModel.from_pretrained("facebook/hubert-base-ls960", trust_remote_code=True, cache_dir=cache_dir_hub, local_files_only=True, device_map=device)

if modtype=='wtv_kr':
    print('FE load : loading wav2vec2-base-korean')
    femod = AutoModel.from_pretrained("eunyounglee/wav2vec_korean", trust_remote_code=True,cache_dir=cache_dir_wv2, local_files_only=True, device_map=device)
#from transformers import AutoProcessor, AutoModelForCTC


#pipe = pipeline("feature-extraction", model="team-lucid/hubert-base-korean", cache_dir=cache_dir_hub)
#pipe = pipeline("feature-extraction", model="facebook/hubert-base-ls960", cache_dir=cache_dir_hub)
# except Exception as e:
#     print('Exception', e, 'from FEload.')
#     is_green=False



# processor = AutoProcessor.from_pretrained("team-lucid/hubert-base-korean", trust_remote_code=True, cache_dir=cache_dir_hub)
#processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=cache_dir_wv2, local_files_only=True)
#processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960", cache_dir=cache_dir_hub)
#processor = AutoProcessor.from_pretrained("Kkonjeong/wav2vec2-base-korean", cache_dir=cache_dir_wv2)