import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import os
from datetime import datetime
from decimal import Decimal
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
import random
import joblib

def optimize_rf_model(X, y, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1):
    estimator = RandomForestRegressor()
    grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    cv=cv,
    scoring=scoring,
    n_jobs = n_jobs,
    verbose=3
    )
    grid_search.fit(X,y)
    print(f'Best parameters found: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_}')
    return grid_search.best_estimator_

hyperparameter_grid = {
    'n_estimators': [50],
    'max_depth': [10, 25],
    'max_features': ['log2'],
    'min_samples_leaf': [2, 3, 4],
    'min_samples_split': [2, 3, 4],
    'oob_score': [True],
    'criterion': ['mse', 'mae'],
    'max_leaf_nodes': [10, 20, 30],
    'min_impurity_decrease': [0.0, 0.1]
}
train_params = ['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']

DataLocation = os.path.join(os.path.dirname(__file__), '..', 'data')
os.chdir(DataLocation)

df = pd.read_csv('dat07_u.csv', low_memory=False).sample(frac=1, random_state=42)
df.dropna(subset=['Rain mm/y', 'koppen_geiger', 'PET mm/y', 'distance_to_coast_km', 'Aridity', 'elevation_mahd', 'wtd_mbgs', 'regolith_depth_mbgs', 'slope_perc', 'clay_perc', 'silt_perc', 'sand_perc', 'soil_class', 'geology', 'ndvi_avg', 'vegex_cat', 'rainfall_seasonality'], inplace=True)
X = df[train_params]
y = df['Recharge RC 50% mm/y']

start_time = datetime.now()

best_model = optimize_rf_model(X, y, hyperparameter_grid)
print(f'Optimized model: {best_model}')
print(f'Time taken: {datetime.now() - start_time}')
