import sklearn
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
import os

print(joblib.__version__)

X_train = pd.read_csv('data/processed/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Load best parameters from gridsearch
best_params = joblib.load('./models/xgb_best_params.pkl')
xgb_regressor = xgb.XGBRegressor(n_jobs=-1, **best_params)

#--Train the model
xgb_regressor.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = './models/trained_model.joblib'
joblib.dump(xgb_regressor, model_filename)
print("Model trained and saved successfully.")
