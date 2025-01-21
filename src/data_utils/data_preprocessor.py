import pandas as pd

class DataPreprocessor:
    def __init__(self) -> None:
        pass

    def to_onehot(self, df: pd.DataFrame, col: str):
        """
        A function converts entries in a column to one-hot vectors.
        Args:
            - df: The dataframe to transform
            - col: The column for which one-hot encoding should be applied.
        """
        df = df.copy()
        df_ohe = pd.get_dummies(df[col], prefix=col)
        df = df.drop(columns=[col])
        df_out = pd.concat([df, df_ohe], axis=1)
        return df_out

    def stratified_split(self, train_fraction: float):
        assert train_fraction < 1, "train_fraction must be less than 1."
        

    def preprocess(self, df: pd.DataFrame):
        
        # convert columns to one-hot vectors
        ohe_columns = ['region', 'NAICS_descr']
        for column in ohe_columns:
            df = self.to_onehot(df, column)
        
        return df