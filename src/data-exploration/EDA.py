from src.utils.load_data import DataLoader

data_loader = DataLoader()

dataset = data_loader.load_dataset("Paper1_WebData_Final.csv")

print(dataset)