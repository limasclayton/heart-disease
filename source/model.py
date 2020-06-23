# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report

# variabls
RANDOM_STATE = 42

# paths
path = "input/heart_disease_dataset.csv"

# reading the dataset
df = pd.read_csv(path)

# changing no heart disease to heart disease for interpretability
df['heart_disease'] = [0 if n else 1 for n in df.num]
df.drop('num', axis=1, inplace=True)

# PROPROCESSING

# FEATURE SELECTION
# selected features duo to EDA
features_1 = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'cp_1', 'cp_2', 'cp_3', 'cp_4', 'restecg_0', 'restecg_1', 'restecg_2', 'slope_1', 'slope_2', 'slope_3', 'thal_3', 'thal_6', 'thal_7', 'ca_0', 'ca_1', 'ca_2', 'ca_3']

features_2 = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'cp_1', 'cp_2', 'cp_3', 'cp_4', 'restecg_0', 'restecg_1', 'slope_1', 'slope_3', 'thal_3', 'thal_6', 'ca_0', 'ca_1', 'ca_2', 'ca_3']

# Getting one hot encoded variables
categorical = ['cp', 'restecg', 'slope', 'thal', 'ca']
df_ohe = pd.get_dummies(df, columns=categorical)

X = df_ohe[features_2]
#X = df_ohe.drop('heart_disease', axis=1)
y = df_ohe.heart_disease
X_train, X_test, y_train, y_test = train_test_split(X, y==1, stratify=y, test_size=0.1, random_state=RANDOM_STATE)
print(X_train.shape, X_test.shape)

# Catboost Feature Selection
cat_features = ['sex', 'fbs', 'exang','cp', 'restecg', 'slope', 'thal', 'ca']
X_cat = df.drop('heart_disease', axis=1)
y_cat = df.heart_disease
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y_cat, test_size=0.1, random_state=RANDOM_STATE)
print(X_train_cat.shape, X_test.shape)

# MODEL

# catboost
cat = CatBoostClassifier(cat_features=cat_features, random_seed=RANDOM_STATE)
#cat.load_model('cat')

cat.fit(X_train_cat, y_train_cat)
print('CatBoost train score: {:.3f}'.format(cat.score(X_train_cat, y_train_cat)))
print('CatBoost test score: {:.3f}'.format(cat.score(X_test_cat, y_test_cat)))
print(classification_report(y_test_cat, cat.predict(X_test_cat)))
#cat.save_model('cat',pool=X_train_cat)

# logistic regression with standard scaler 
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
    ])

param_grid = [{
    'classifier__penalty' : ['l1'],
    'classifier__solver' : ['liblinear'],
    'classifier__C' : np.logspace(-3, 1, 20),
    'classifier__max_iter' : np.arange(100, 400, 50),
    'classifier__random_state' : [RANDOM_STATE]
    }]

lr_cv = RandomizedSearchCV(lr_pipe, param_grid, n_jobs=-1, cv=10, random_state=RANDOM_STATE)
lr_cv.fit(X_train, y_train)

print('-' * 100)
print('Logistic Regression pipeline train score: {:.3f}'.format(lr_cv.score(X_train, y_train)))
print('Logistic Regression pipeline test score: {:.3f}'.format(lr_cv.score(X_test, y_test)))
print('Logistic Regression pipelinepeline Best score: {0}'.format(lr_cv.best_score_))
print('Logistic Regression pipeline best params: {0}'.format(lr_cv.best_params_))
print('Logistic Regression pipeline coeficients: {0}'.format(lr_cv.best_estimator_.named_steps['classifier'].coef_))

# xgboost
param_distributions_xgb = {
    'learning_rate' : np.linspace(0, 1, 50),
    'min_split_loss' : np.logspace(1, 3, 10),
    'max_depth' : np.arange(2, 10, 1),
    'min_child_weight' : np.arange(0, 5, 1),
    'subsample' : np.linspace(0.01, 1, 10),
    'colsample_bytree' : np.linspace(0.01, 1, 10),
    'colsample_bylevel' : np.linspace(0.01, 1, 10),
    'colsample_bynode' : np.linspace(0.01, 1, 10),
    'random_state' : [RANDOM_STATE]
}

xgb = XGBClassifier()
xgb_CV = RandomizedSearchCV(xgb, param_distributions=param_distributions_xgb, cv=10, n_jobs=-1, random_state=RANDOM_STATE)
xgb_CV.fit(X_train, y_train)
print('-' * 100)
print('XGB train score: {:.3f}'.format(xgb_CV.score(X_train, y_train)))
print('XGB test score: {:.3f}'.format(xgb_CV.score(X_test, y_test)))
print('XGB best params: {0}'.format(xgb_CV.best_params_))
print(classification_report(y_test, xgb_CV.predict(X_test)))