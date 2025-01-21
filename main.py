import pandas as pd
import matplotlib.pyplot as plt
import pdb

from src.data_utils.utils import load_csv
from src.data_utils.data_preprocessor import DataPreprocessor

data_dir = 'data/sd-1001-2014-0/CSV'
df = load_csv(filepath=f'{data_dir}/Paper1_WebData_Final.csv')

# data_preprocessor = DataPreprocessor()
# df = data_preprocessor.preprocess(df)

ag_hist = df['age_group'].hist()

pdb.set_trace()


