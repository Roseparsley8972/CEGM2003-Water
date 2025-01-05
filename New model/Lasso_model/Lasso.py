import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from datetime import datetime
from decimal import Decimal
import random
import joblib
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


def run_lasso_model(t_size=0.3, alpha = 0.1, k_num=10, y_var='Recharge RC 50% mm/y', y_predict='R50', aus_file='Australia_grid_0p05_data.csv', seed=42, test_data=True, tuning = True):

    
    #alpha: A regularization parameter, affects flexibility 
    start_time = datetime.now()
    DataLocation = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.chdir(DataLocation)
    random.seed(seed)
    random_num = random.randint(0, 1000)

    train_params = ['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']
    
    #The file that is used to create the map
    aus_X = pd.read_csv(aus_file)[train_params]

    if not test_data:
        df = pd.read_csv('dat07_u.csv', low_memory=False).sample(frac=1, random_state=seed)
        df.dropna(subset=['Rain mm/y', 'koppen_geiger', 'PET mm/y', 'distance_to_coast_km', 'Aridity', 'elevation_mahd', 'wtd_mbgs', 'regolith_depth_mbgs', 'slope_perc', 'clay_perc', 'silt_perc', 'sand_perc', 'soil_class', 'geology', 'ndvi_avg', 'vegex_cat', 'rainfall_seasonality'], inplace=False)
        print(f"nans removed, removed {len(df) - len(df.dropna())}, removed {Decimal(100 * (len(df) - len(df.dropna()))/len(df)).quantize(Decimal('1.0'))}%")
        print(f"Remaining data has mean Rrc/P ratio: {Decimal(np.nanmean(df['Rrc/P'])).quantize(Decimal('1.00'))}")
        X = df[train_params]
        y = df[y_var]
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=t_size, random_state=random_num)

    else:
        train_data_file = 'train_data.csv'
        train_data = pd.read_csv(train_data_file)

        # Assign features and target variable
        X = train_data[train_params]
        y = train_data[y_var]

        # Split the data into training and validation sets
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=t_size, random_state=random_num)

    
    #Hyperparameter tuning
    #The best value is 0.0002 according to the tuning, but does not improve the model
    if tuning:
        model = Lasso()
        param_grid = {'alpha': np.logspace(-6, 1, 10)}  # Test alpha values from 0.000001 to 10

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=k_num, n_jobs=-1)
        
        # Fit the estimator to find the best parameters
        grid_search.fit(Xtrain, ytrain)

        # Get the best estimator and its parameters
        best_lasso = grid_search.best_estimator_
        best_alpha = grid_search.best_params_['alpha']

        print(f'Best alpha: {best_alpha:.4f}')
        print(f'Training Score (Best Model): {best_lasso.score(Xtrain, ytrain):.3f}')
        print(f'Validation Score (Best Model): {best_lasso.score(Xvalid, yvalid):.3f}')
        
        # Make predictions using the best model
        ypredv = best_lasso.predict(Xvalid)
    else:
        lasso = Lasso(alpha=alpha)
        lasso.fit(Xtrain, ytrain)
        ypredv = lasso.predict(Xvalid)

    #Selects the correct lasso model based on tuning or not
    lasso = best_lasso if tuning else lasso
    print(f'Training Score: {lasso.score(Xtrain, ytrain):.3f}')
    print(f'Validation Score: {lasso.score(Xvalid, yvalid):.3f}')

    # Validation
    y_pred_valid = pd.DataFrame({'y_predict': ypredv, 'y_validation': yvalid})
    y_pred_valid['Residual (predicted R - CMB R)'] = y_pred_valid['y_predict'] - y_pred_valid['y_validation']
    y_pred_valid['Residual (%)'] = ((y_pred_valid['y_predict'] - y_pred_valid['y_validation']) / y_pred_valid['y_validation']) * 100

    print(f'R2 Score: {r2_score(yvalid, ypredv):.3f}')
    print(f'RMSE: {mean_squared_error(yvalid, ypredv, squared=False):.3f}')
    print(f'MAE: {mean_absolute_error(yvalid, ypredv):.3f}')

    # Cross-validation
    scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}
    cv_results = cross_validate(lasso, Xtrain, ytrain, cv=k_num, scoring=scoring, n_jobs=-1)
    print(f'k={k_num}')
    print(f'R2 Score: {np.mean(cv_results["test_r2"]):.3f}')
    print(f'RMSE: {-np.mean(cv_results["test_rmse"]):.1f}')
    print(f'MAE: {-np.mean(cv_results["test_mae"]):.1f}')

    print('Starting predictions...')
    y_pred_aus = pd.DataFrame({'lat': pd.read_csv(aus_file).iloc[:,0], 'lon': pd.read_csv(aus_file).iloc[:,1], y_predict: lasso.predict(aus_X)})
    
    print('Finished prediction... writing values')

    y_pred_valid.to_csv(f'model_validation_predictions_errors_lasso_alpha{alpha}_{k_num}fold_out.csv', index=False)
    y_pred_aus.to_csv(f'model_predictions_aus_lasso_alpha{alpha}_{k_num}fold_out.csv', index=False)
    print(f'Model took: {(datetime.now() - start_time).total_seconds()/60:.2f} minutes to run')

if __name__ == "__main__":
    run_lasso_model()