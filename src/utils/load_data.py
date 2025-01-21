import os
from pathlib import Path
from typing import Dict, Union, List, Tuple
import numpy as np
import pandas as pd





# class DataLoader:
#     """Data loader class for handling dataset loading and preprocessing."""
    
#     def __init__(self, data_dir: Union[str, Path] = None):
#         """
#         Initialize the data loader.
        
#         Args:
#             data_dir: Path to the data directory. If None, uses the project root's data/CSV directory.
#         """
#         if data_dir is None:
#             # Get the project root directory (where src is located)
#             project_root = Path(__file__).parent.parent.parent
#             data_dir = project_root / "data" / "CSV"
        
#         self.data_dir = Path(data_dir)
#         if not self.data_dir.exists():
#             raise FileNotFoundError(
#                 f"Data directory {self.data_dir} does not exist. "
#                 f"Please create a 'CSV' directory in {self.data_dir.parent}"
#             )
    
#     def load_dataset(self, dataset_name: str) -> pd.DataFrame:
#         """
#         Load a dataset from the data directory.
        
#         Args:
#             dataset_name: Name of the dataset file (without extension)
            
#         Returns:
#             pd.DataFrame: Loaded dataset as a pandas DataFrame
        
#         Raises:
#             FileNotFoundError: If the dataset file doesn't exist
#             ValueError: If there's an error loading the dataset
#         """
#         # Remove .csv extension if it was included
#         dataset_name = dataset_name.replace('.csv', '')
        
#         # Check for CSV file
#         csv_path = self.data_dir / f"{dataset_name}.csv"
        
#         if not csv_path.exists():
#             raise FileNotFoundError(
#                 f"Dataset file {csv_path} does not exist. "
#                 f"Available files in {self.data_dir}: {[f.name for f in self.data_dir.glob('*.csv')]}"
#             )
        
#         try:
#             # Load CSV file into pandas DataFrame
#             df = pd.read_csv(csv_path)
#             return df
#         except Exception as e:
#             raise ValueError(f"Error loading dataset {csv_path}: {str(e)}")
            