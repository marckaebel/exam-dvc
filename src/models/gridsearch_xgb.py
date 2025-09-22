import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import os

# Load data
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_train = np.ravel(y_train)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_regressor = xgb.XGBRegressor(n_jobs=-1)

grid_search = GridSearchCV(estimator=xgb_regressor,
                           param_grid=param_grid,
                           cv=3,
                           scoring='neg_mean_squared_error',
                           verbose=1,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

# Save best parameters
os.makedirs('./models', exist_ok=True)
joblib.dump(best_params, './models/xgb_best_params.pkl')
print("Best parameters saved to ./models/xgb_best_params.pkl")
