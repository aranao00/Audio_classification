import subprocess
import json

# 하이퍼 파라미터 튜닝 자동화용 코드입니다.

config_path='/tf/nasw/config.json'
script_path='/tf/nasw/Train_PDclf.py'


{
       "__FEmodelSAMPLE__": "hub_kr, wtv_kr, hub_base, wtv_base",
       "__CLFmodelSAMPLE__": "self-attention, cross-attention, linear",
       "FEmodel": "wtv_base",
       "CLFmodel": "linear",
       "self-attention_config": {
              "d_model": 6,
              "layers": 3,
              "nhead": 3
       },
       "cross-attention_config": {
              "hdim": 256,
              "d_model": 6,
              "layers": 3,
              "nhead": 3
       },
       "linear_config": {},
       "optim": "Adam",
       "batchsize": 64,
       "lr": 2e-05,
       "epochs": 300,
       "iter_per_epochs": 3,
       "loss_fn": "BCE",
       "device": "cuda",
       "task_number": "007",
       "seed": 42,
       "debug": True,
       "duration": 6
}

tgttask=["006", "007", "008", "009", "010", "011", "012", "002"]
femods=['wtv_kr', 'wtv_base']#'hub_kr', 'wtv_base']#'wtv_base', 'hub_base', 'hub_kr', 'wtv_base']

def run_script(script_path):
    subprocess.run(['python3', script_path])

def update_config(tgtkey, tgtval):
    with open(config_path, 'r') as file:
        config=json.load(file)
    config[tgtkey]=tgtval
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=7)
  
if __name__=="__main__":
    update_config('batchsize', 64)
    # update_config('CLFmodel', 'cross-attention')
    # update_config('epochs', 300)
    # for femod in femods:
    #     update_config('FEmodel', femod)
    #     for tasks in tgttask:
    #         update_config('task_number', tasks)
    #         run_script(script_path)
            

    update_config('epochs', 300)
    update_config('CLFmodel', 'linear')
    for femod in femods:
        update_config('FEmodel', femod)
        for tasks in tgttask:
            update_config('task_number', tasks)
            run_script(script_path)
    