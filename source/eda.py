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
categorical = ['cp', 'restecg', 'slope', 'thal', 'ca']

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

# Is the dataset balanced? 164/139, more or less
print(df.heart_disease.value_counts())

# Noticed -100000 in thal, how many are there? Only 4, keep for now, but as cat
print(df.thal.value_counts())

# Noticed -100000 in ca, how many are there?
print(df.ca.value_counts())

# How are the variables related to the target?
corr = df_ohe.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, fmt='.1f')
plt.show()

# Features with absolute correlation > selected threshold to target (heart_disease)
threshold = 0.05 # low threshold to lose little variance as possible
print(corr)
print(corr.heart_disease[abs(corr.heart_disease) > threshold])
features = corr.heart_disease[abs(corr.heart_disease) > threshold].index.values
print(features)

# Features with high correlation between themselves
# restecg_2 and restecg_0: -1.0
# slope_1 and slope_2: -0.9
# thal_3 and thal_7: -0.9

# Observations
# sex, cp, fbs, restecg, exang, slope, thal are cat and not int
# scale trestbps, chol, thalach, oldpeak

# Selected Features
# All
# First selection: only removing threshold
# ['age' 'sex' 'trestbps' 'chol' 'thalach' 'exang' 'oldpeak' 'heart_disease'  'cp_1' 'cp_2' 'cp_3' 'cp_4' 'restecg_0' 'restecg_1' 'restecg_2' 'slope_1' 'slope_2' 'slope_3' 'thal_3' 'thal_6' 'thal_7' 'ca_0' 'ca_1' 'ca_2' 'ca_3']
# Second selection: removing threshold and high correlated features between themselves
# ['age' 'sex' 'trestbps' 'chol' 'thalach' 'exang' 'oldpeak' 'heart_disease'  'cp_1' 'cp_2' 'cp_3' 'cp_4' 'restecg_0' 'restecg_1' 'slope_1' 'slope_3' 'thal_3' 'thal_6' 'ca_0' 'ca_1' 'ca_2' 'ca_3']
