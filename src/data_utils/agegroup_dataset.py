import torch
import pandas as pd
from torch.utils.data import Dataset

class AgegroupDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df
        X_columns = [e for e in df.columns if e != 'age_group']
        self.X = torch.from_numpy(df[X_columns].values)
        self.y = torch.from_numpy(df['age_group'].values)
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx].unsqueeze(0)
    
    def __len__(self):
        return len(self.df)