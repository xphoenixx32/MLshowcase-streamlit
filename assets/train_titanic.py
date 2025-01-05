# train_titanic.py

import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df = sns.load_dataset('titanic')

# 0) 前處理
valid_values = ["yes", "no"]
df = df[df["alive"].isin(valid_values)]
df = df.dropna(subset=["alive"])
df['alive'] = df['alive'].map({'yes': 1, 'no': 0})

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna('Unknown')
df['family_size'] = df['sibsp'] + df['parch'] + 1
df = df.drop(columns = ['deck'])

columns_to_drop = ['adult_male', 'who', 'survived', 'deck', 'embarked', 'pclass', 'alone', 'deck']
df.drop(columns = [col for col in columns_to_drop if col in df.columns], inplace = True)
df.dropna(axis = 0, how = "any")

# 1) Feature 選取
y = df["alive"]
X = df.drop(columns = ["alive"])

X = pd.get_dummies(X, drop_first = True)

# 2) 建立模型 + GridSearch
model = RandomForestClassifier(random_state = 42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 7, 10, 12]
}
grid_search = GridSearchCV(model, param_grid, cv = 5, n_jobs = -1)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_

# 3) 計算 SHAP values
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X)  # (n_classes, n_samples, n_features) 對於分類

# 4) 儲存檔案
with open("titanic_best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("titanic_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

# shap_values 是多維結構（對於二元分類，形狀如 shap_values[0], shap_values[1]）
np.save("titanic_shap_values.npy", shap_values)

with open("titanic_best_params.pkl", "wb") as f:
    pickle.dump(grid_search.best_params_, f)