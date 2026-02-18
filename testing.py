import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import matplotlib.pyplot as plt
import time
from statsmodels.miscmodels.ordinal_model import OrderedModel



#x  = torch.tensor([2.5, 0.1])
#print(x ,"\n \n")

print("""                                                             
 ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄     ▄▄▄▄   ▄▄▄      ▄▄▄     ▄▄▄▄     ▄▄▄▄   
███▀▀▀▀▀ ███▀▀███▄ ▄██▀▀██▄ ████▄  ▄████   ▄█████   ▄██████▄ 
███      ███▄▄███▀ ███  ███ ███▀████▀███      ███   ███  ███ 
███      ███▀▀██▄  ███▀▀███ ███  ▀▀  ███      ███   ███▄▄███ 
▀███████ ███  ▀███ ███  ███ ███      ███      ███ ██ ▀████▀  
                                                             
                                                             """)

leukemia_data = "Leukemia_GSE9476.csv"
df = pd.read_csv(leukemia_data)

# LOAD AND READ DATA WITH PD FROM PANDA

data = df.head(5) 

# creates the rows to establish the data frame 

print(f"Overall data: \n \n") 
print(data) 

#prints the data and makes it organized

if df.columns[0].lower() in ["id", "sample", "unnamed: 0"]:
    df = df.iloc[:, 1:]

matrix_of_data = df.iloc[:3, :6] 

'''
[:y :x] creates the tensor 
matrix of data by slicing the data frame, 
the first 3 rows and the first 6 columns
'''

print(f"Matrix subset of Dataset: \n \n")
print(matrix_of_data)




data['sample1'].dtypes


mod_ord = OrderedModel()