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

# df infos
print(df.head())

#Is there missing values?
print(df.info())

# Separating categorical variables
categorical = ['sex', 'fbs', 'exang', 'cp', 'restecg', 'slope', 'thal', 'ca']

# Getting one hot encoded variables
df_ohe = pd.get_dummies(df, columns=categorical)
print(df_ohe.info())

# Separating numerical variables to take a look and scale after
numerical = ['trestbps', 'chol', 'thalach', 'oldpeak']

# Look at numeric variables stats
for n in numerical:
    print()
    print('Feature:', n)
    print(df[n].describe())

# Noticed -100000 in thal, how many are there? Only 4, keep for now, but as cat
print(df.thal.value_counts())

# Noticed -100000 in ca, how many are there?
print(df.ca.value_counts())

# How are the variables related to the target?

# Is the dataset balanced? 164/139
print(df.heart_disease.value_counts())

# Observations
# sex, cp, fbs, restecg, exang, slope, thal are cat and not int
# scale trestbps, chol, thalach, oldpeak