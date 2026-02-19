import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm

from sklearn.linear_model import Ridge

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


print('\n \n \n')


X = df.iloc[:, 2:10]   # first 8 gene columns
y = df['type'].astype('category').cat.codes


# attempting ridge regression model to predict the effect leukemia has on gene expression levels

X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()

print(results.summary())


model = Ridge(alpha=1.0)
model.fit(X, y)

print(model.coef_)

for gene, coef in zip(X.columns, model.coef_):
    print(gene, coef)