import json
import os
import pandas as pd
import numpy as np
import torch
import pdb
root_dir = '/work3/s204138/MachineUnlearning/results'
dataset_name = 'MNIST'

def mean_std(x):
    mean = np.mean(x)
    std = np.std(x)
    return f"{mean:.3f} ± {std:.3f}"

def get_all_results():
    agg_cols = ['MIA-probability', 'retain-accuracy', 'forget-accuracy', 'val-accuracy', 'time (sec)']
    result = None
    seeds = os.listdir(f'{root_dir}/{dataset_name}')
    for i, seed in enumerate(seeds):
        with open(f'{root_dir}/{dataset_name}/{seed}/all_results.json', 'r') as f:
            seed_result_dict = json.load(f)
        seed_result_df = pd.DataFrame.from_dict(seed_result_dict)
        if i == 0:
            result = seed_result_df
        else:
            result = pd.concat([result, seed_result_df], ignore_index=True)
    result['hyperparameters'] = result['hyperparameters'].apply(lambda x: str(x))
    agg_result = result.groupby(['hyperparameters', 'model-name'])[agg_cols].agg({col: mean_std for col in agg_cols})
    # print(agg_result['model-name', 'MIA-probability', 'retain-accuracy', 'forget-accuracy', 'val-accuracy', 'time (sec)'])
    pdb.set_trace()
        
get_all_results()