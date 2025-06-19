import json
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pdb
from prepare_image_data import get_mnist_unlearn_data

root_dir = '/work3/s204138/MachineUnlearning/results'
dataset_name = 'MNIST'

def mean_std(x):
    mean = np.mean(x)
    std = np.std(x)
    return f"{mean:.2f} ± {std:.2f}"

def get_all_results():
    agg_cols = ['MIA-probability', 'retain-accuracy', 'forget-accuracy', 'val-accuracy', 'time (sec)']
    result = None
    seeds = os.listdir(f'{root_dir}/{dataset_name}')
    for i, seed in enumerate(seeds):
        with open(f'{root_dir}/{dataset_name}/{seed}/all_results.json', 'r') as f:
            seed_result_dict = json.load(f)
        seed_result_df = pd.DataFrame.from_dict(seed_result_dict)
        
        ssd_smooth_idx = seed_result_df.loc[seed_result_df['hyperparameters'] == {'P': 3, 'k': 0.75, 'smooth_dampening': True, 'n_bo_iter': 20}].index
        seed_result_df.loc[ssd_smooth_idx, 'model-name'] = 'SSD v6 smooth'
        amnesiac_repair_idx = seed_result_df.loc[seed_result_df['hyperparameters'] == {'repair': True, 'n_repair_epochs': 3}].index
        seed_result_df.loc[amnesiac_repair_idx, 'model-name'] = 'Amnesiac repair'

        if i == 0:
            result = seed_result_df
        else:
            result = pd.concat([result, seed_result_df], ignore_index=True)
    result['hyperparameters'] = result['hyperparameters'].apply(lambda x: str(x))
    agg_result = result.groupby(['hyperparameters', 'model-name'])[agg_cols].agg({col: mean_std for col in agg_cols})
    
    # print(agg_result['model-name', 'MIA-probability', 'retain-accuracy', 'forget-accuracy', 'val-accuracy', 'time (sec)'])

# def make_latex_table()

def get_performance_result(seed: int):
    with open(f'{root_dir}/{dataset_name}/seed_{seed}/all_results.json', 'r') as f:
        seed_result_dict = json.load(f)
    seed_result_df = pd.DataFrame.from_dict(seed_result_dict)


def get_wrong_preds(seed: int):
    with open(f'{root_dir}/{dataset_name}/seed_{seed}/wrong_forget_preds.json', 'r') as f:
        wrong_preds = json.load(f)
    # df = pd.DataFrame.from_dict(wrong_preds)
    return wrong_preds
    

def get_train_data(seed: int):
    dataloader_train, dataloader_retain, dataloader_forget, dataloader_val, forget_idxs = get_mnist_unlearn_data('/work3/s204138/MachineUnlearning/data', 
                                                                                                                 'MNIST', n_forget_points=100, 
                                                                                                                 subsample_size=10000, 
                                                                                                                 batch_size=16, seed=seed)
    return dataloader_train#, forget_idxs


def save_wrong_pred_images(seed: int):
    import matplotlib.pyplot as plt
    wrong_preds = get_wrong_preds(seed)
    dataloader_train = get_train_data(seed)
    X = dataloader_train.dataset.X
    
    ssd_v6_seen = False
    amnesiac_seen = False
    
    for i, model_name in enumerate(wrong_preds['model-name']):
        wrong_idxs = wrong_preds['wrong-preds'][i]
        wrong_imgs = X[wrong_idxs, :]

        if model_name == 'Amnesiac' and not amnesiac_seen:
            amnesiac_seen = True
            amnesiac_idx = i
        
        elif model_name == 'SSD v6' and not ssd_v6_seen:
            ssd_v6_seen = True
            ssd_v6_idx = i

        for idx in range(wrong_imgs.size(0)):
            save_dir = f'exp4/results/wrong_preds/seed_{seed}/{model_name}'
            
            if model_name == 'SSD v6' and ssd_v6_seen and i > ssd_v6_idx:
                save_dir = f'exp4/results/wrong_preds/seed_{seed}/{model_name}-smooth'
            
            if model_name == 'Amnesiac' and amnesiac_seen and i > amnesiac_idx:
                save_dir = f'exp4/results/wrong_preds/seed_{seed}/{model_name}-repair'
            
            os.makedirs(save_dir, exist_ok=True)
            img = wrong_imgs[idx].view(28, 28)
            plt.imsave(fname=f'{save_dir}/img_{wrong_idxs[idx]}.png', arr=img.numpy(), cmap='grey')

get_all_results()
# save_wrong_pred_images(3)
# get_wrong_preds(seed=6)

# get_performance_result(6)
# cols = ['model-name', 'MIA-probability', 'retain-accuracy', 'forget-accuracy','val-accuracy', 'time (sec)']