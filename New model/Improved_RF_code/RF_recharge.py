import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from scipy.stats import randint
# from skopt import BayesSearchCV

import os
from datetime import datetime
from decimal import Decimal
import random
import joblib

def run_rf_model(t_size=0.3, trees=50, max_splits=40, max_features='log2', min_samples_leaf=8, min_samples_split=8, k_num=10, y_var='Recharge RC 50% mm/y', y_predict='R50', aus_file='Australia_grid_0p05_data.csv', seed=42, test_data=False):
    start_time = datetime.now()
    DataLocation = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.chdir(DataLocation)

    df = pd.read_csv('dat07_u.csv', low_memory=False).sample(frac=1, random_state=seed)
    df.dropna(subset=['Rain mm/y', 'koppen_geiger', 'PET mm/y', 'distance_to_coast_km', 'Aridity', 'elevation_mahd', 'wtd_mbgs', 'regolith_depth_mbgs', 'slope_perc', 'clay_perc', 'silt_perc', 'sand_perc', 'soil_class', 'geology', 'ndvi_avg', 'vegex_cat', 'rainfall_seasonality'], inplace=True)
    print(f"nans removed, removed {len(df) - len(df.dropna())}, removed {Decimal(100 * (len(df) - len(df.dropna()))/len(df)).quantize(Decimal('1.0'))}%")
    print(f"Remaining data has mean Rrc/P ratio: {Decimal(np.nanmean(df['Rrc/P'])).quantize(Decimal('1.00'))}")

    train_params = ['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']

    aus_X = pd.read_csv(aus_file)[train_params]
    random.seed(seed)
    random_num = random.randint(0, 1000)

    if not test_data:
        X = df[train_params]
        y = df[y_var]
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=t_size, random_state=random_num)

    else:
        train_data_file = 'train_data.csv'

        train_data = pd.read_csv(train_data_file)
        Xtrain = train_data[['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']]
        ytrain = train_data[y_var]

    rf = RandomForestRegressor(n_estimators=trees, max_depth=max_splits, random_state=random_num, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, bootstrap=False, oob_score=False)
    rf.fit(Xtrain, ytrain)

    print(f'Training Score: {rf.score(Xtrain, ytrain):.3f}')

    if not test_data:
        ypredv = rf.predict(Xvalid)
        y_pred_valid = pd.DataFrame({'y_predict': ypredv, 'y_validation': yvalid})
        y_pred_valid['Residual (predicted R - CMB R)'] = y_pred_valid['y_predict'] - y_pred_valid['y_validation']
        y_pred_valid['Residual (%)'] = ((y_pred_valid['y_predict'] - y_pred_valid['y_validation']) / y_pred_valid['y_validation']) * 100
        print(f'R2 Score: {r2_score(yvalid, ypredv):.3f}')
        print(f'RMSE: {mean_squared_error(yvalid, ypredv, squared=False):.3f}')
        print(f'MAE: {mean_absolute_error(yvalid, ypredv):.3f}')

        scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}
        cv_results = cross_validate(rf, Xtrain, ytrain, cv=k_num, scoring=scoring, n_jobs=-1)
        print(f'k={k_num}')
        print(f'R2 Score: {np.mean(cv_results["test_r2"]):.3f}')
        print(f'RMSE: {-np.mean(cv_results["test_rmse"]):.1f}')
        print(f'MAE: {-np.mean(cv_results["test_mae"]):.1f}')

        print('Starting predictions...')
        y_pred_aus = pd.DataFrame({'lat': pd.read_csv(aus_file).iloc[:,0], 'lon': pd.read_csv(aus_file).iloc[:,1], y_predict: rf.predict(aus_X)})
        print('Finished prediction... writing values')

        y_pred_valid.to_csv(f'model_validation_predictions_errors_50_{trees}trees_mf{max_features}_{k_num}fold_out.csv', index=False)
        y_pred_aus.to_csv(f'model_predictions_aus_{trees}trees_mf{max_features}_{k_num}fold_out.csv', index=False)
    print(f'Model took: {(datetime.now() - start_time).total_seconds()/60:.2f} minutes to run')

    # Save the trained model to a file if using test data
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'Trained_models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filename = os.path.join(model_dir, f'rf_model_{trees}trees_mf{max_features}_{test_data}.pkl')
    joblib.dump(rf, model_filename)
    print(f'Model saved to {model_filename}')

if __name__ == "__main__":
    run_rf_model(test_data=False)

def optimize_rf_model(X, y, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1):
    estimator = RandomForestRegressor()
    grid_search = HalvingGridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    cv=cv,
    scoring=scoring,
    n_jobs = n_jobs,
    verbose=2
    )
    grid_search.fit(X,y)
    print(f'Best parameters found: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_}')
    return grid_search.best_estimator_

def Bayes_opt_rf(X,y,param_grid):
    bayes_opt =  BayesSearchCV(
    RandomForestRegressor(),
    param_grid,
    n_iter=32,  # Number of parameter settings to try
    cv=3,       # Cross-validation
    n_jobs=-1,  # Use all processors
    random_state=42,
    verbose=2
    )
    bayes_opt.fit(X,y)
    print("Best Hyperparameters:", bayes_opt.best_params_)
    print("Best Score:", bayes_opt.best_score_)
    return bayes_opt.best_estimator_

# hyperparameter_grid = {'n_estimators':[100,200,300,400,500,600,700,800,900,1000],
#                         'max_depth': [3,8,13,18,23,28,33,38,43,48,50],
#                         'min_samples_split': [2,5,10,15,20],
#                         'min_samples_leaf': [1,5,10,15,20,25,30,35,40,45,50],
#                         'max_features': [0.1,0.3,0.5,0.7,0.9]   }

# train_params = ['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']

# df = pd.read_csv('dat07_u.csv', low_memory=False).sample(frac=1, random_state=42)
# df.dropna(subset=['Rain mm/y', 'koppen_geiger', 'PET mm/y', 'distance_to_coast_km', 'Aridity', 'elevation_mahd', 'wtd_mbgs', 'regolith_depth_mbgs', 'slope_perc', 'clay_perc', 'silt_perc', 'sand_perc', 'soil_class', 'geology', 'ndvi_avg', 'vegex_cat', 'rainfall_seasonality'], inplace=True)
# X = df[train_params]
# y = df['Recharge RC 50% mm/y']
# best_model = Bayes_opt_rf(X, y, hyperparameter_grid)
# print(f'Optimized model: {best_model}')