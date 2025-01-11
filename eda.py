import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='train.xlsx')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path

df = pd.read_excel("superstore1.xls")
# To display the top 5 rows 
df.head(5) 