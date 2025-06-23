import json
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pdb
from prepare_image_data import get_mnist_unlearn_data

root_dir = '/work3/s204138/MachineUnlearning/results'
dataset_name = 'MNIST' #'CIFAR10'

def mean_std(x):
    mean = np.mean(x)
    std = np.std(x)
    return f"{mean:.2f} ± {std:.2f}"

def get_all_results():
    agg_cols = ['MIA-probability', 'retain-accuracy', 'forget-accuracy', 'val-accuracy', 'time (sec)']
    result = None
    # seeds = os.listdir(f'{root_dir}/{dataset_name}')
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i, seed in enumerate(seeds):
        with open(f'{root_dir}/{dataset_name}/seed_{seed}/all_results.json', 'r') as f:
            seed_result_dict = json.load(f)
        seed_result_df = pd.DataFrame.from_dict(seed_result_dict)
        
        if i == 0:
            result = seed_result_df
        else:
            result = pd.concat([result, seed_result_df], ignore_index=True)
    result['hyperparameters'] = result['hyperparameters'].apply(lambda x: str(x))
    agg_result = result.groupby(['hyperparameters', 'model-name'])[agg_cols].agg({col: mean_std for col in agg_cols})
    pdb.set_trace()
    # print(agg_result['model-name', 'MIA-probability', 'retain-accuracy', 'forget-accuracy', 'val-accuracy', 'time (sec)'])

# def make_latex_table()

def get_js_divergence_results():
    agg_cols = ['JS div. original',  'JS div. retrained']
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i, seed in enumerate(seeds):
        with open(f'{root_dir}/{dataset_name}/seed_{seed}/JS_divergence_results.json', 'r') as f:
            seed_result_dict = json.load(f)
        seed_result_df = pd.DataFrame.from_dict(seed_result_dict)
        if i == 0:
            result = seed_result_df
        else:
            result = pd.concat([result, seed_result_df], ignore_index=True)
    
    agg_result = result.groupby(['model-name', 'dataset'])[agg_cols].agg({col: mean_std for col in agg_cols})
    pdb.set_trace()
        


def get_wrong_preds(seed: int):
    with open(f'{root_dir}/{dataset_name}/seed_{seed}/wrong_forget_preds.json', 'r') as f:
        wrong_preds = json.load(f)
    # df = pd.DataFrame.from_dict(wrong_preds)
    pdb.set_trace()
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

# get_all_results()
# get_js_divergence_results()

# save_wrong_pred_images(3)
get_wrong_preds(seed=6)
