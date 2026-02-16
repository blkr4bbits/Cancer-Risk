import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import matplotlib.pyplot as plt




x  = torch.tensor([2.5, 0.1])
print(x ,"\n \n")


leukemia_data = "Leukemia_GSE9476.csv"
df = pd.read_csv(leukemia_data)

data = df.head()
print(f"Overall data: \n \n")
print(data)

if df.columns[0].lower() in ["id", "sample", "unnamed: 0"]:
    df = df.iloc[:, 1:]

matrix_of_data = df.iloc[:3, :6] # [:y :x]

print(f"Matrix subset of Dataset: \n \n")
print(matrix_of_data)