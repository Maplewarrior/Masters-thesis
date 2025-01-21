from src.utils.load_data import LoadData
import pdb

data_loader = LoadData()

dataset = LoadData.load_dataset("Paper1_WebData_Final.csv")

print(dataset)

pdb.set_trace()