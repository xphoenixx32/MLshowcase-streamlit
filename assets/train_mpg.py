# train_mpg.py

import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import shap
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay

df = sns.load_dataset('mpg')

# 1) 前處理
X = df.drop(columns=['mpg', 'name'])
X = pd.get_dummies(X, drop_first=True)
y = df['mpg']

# 2) 建立模型 + GridSearch
model = LGBMRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10]
}
grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_

# 3) 計算 SHAP values
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X)  # array shape: (n_samples, n_features)

# 4) 將所需檔案存起來 (模型、SHAP values、或其他資訊)
with open("mpg_best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("mpg_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

# shap_values 通常是 numpy array，可用 np.save 或直接用 pickle
np.save("mpg_shap_values.npy", shap_values)

# (若需要儲存 grid_search.best_params_ 也可另外存)
with open("mpg_best_params.pkl", "wb") as f:
    pickle.dump(grid_search.best_params_, f)