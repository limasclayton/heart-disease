# Heart Disease

### Input
Instances: 303

Attributes: 14

Year Published: 1988

Link: [heart_disease](https://www.mldata.io/dataset-details/heart_disease/)


### Objective
Predict Heart Disease in Patients from Cleveland

### Output
source/EDA.py has the exploratory data analysis done in the dataset, feature selection based on the correlation between features and target and features and themselves.

source/model.py has the feature selection extrated on the EDA and the three models applied to them. Logistic Regression, CatBoost and XGboost.

source/models.csv has the chosen models, parameters and their detalied scores.
