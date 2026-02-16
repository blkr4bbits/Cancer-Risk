import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import matplotlib.pyplot as plt


x  = torch.tensor([2.5, 0.1])
print(x)

file_path = r"C:\Users\Joshua Green\OneDrive\Desktop\Leukemia_GSE9476.csv"

df = pd.read_csv(file_path)

print("First 5 records:", df.head())

y = torch.tensor(df.values, dtype=torch.float32)

print("Tensor shape:", y.shape)