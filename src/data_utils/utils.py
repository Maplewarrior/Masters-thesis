import os
import pandas as pd

def load_csv(filepath: str):
    """
    A function that loads a csv file from the specified filepath
    """
    assert os.path.exists(filepath), 'The specified filepath does not exist.'
    df = pd.read_csv(filepath)
    return df