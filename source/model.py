# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# paths
path = "input/heart_disease_dataset.csv"

# reading the dataset
df = pd.read_csv(path)

# changing no heart disease to heart disease for interpretability
df['heart_disease'] = [0 if n else 1 for n in df.num]
df.drop('num', axis=1, inplace=True)