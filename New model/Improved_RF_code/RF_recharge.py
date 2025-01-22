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

def run_rf_model(t_size=0.3, trees=250, max_splits=18, max_features=0.33, min_samples_leaf=8, min_samples_split=2, k_num=10, y_var='Recharge RC 50% mm/y', y_predict='R50', aus_file='Australia_grid_0p05_data.csv', seed=42, test_data=False):
    start_time = datetime.now()
    DataLocation = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.chdir(DataLocation)

    features = ['Rain mm/y', 'lat', 'lon', 'koppen_geiger', 'PET mm/y', 'distance_to_coast_km', 'Aridity', 'elevation_mahd', 'wtd_mbgs', 'regolith_depth_mbgs', 'slope_perc', 'clay_perc', 'silt_perc', 'sand_perc', 'soil_class', 'geology', 'ndvi_avg', 'vegex_cat', 'rainfall_seasonality']

    df = pd.read_csv('dat07_u.csv', low_memory=False).sample(frac=1, random_state=seed)
    df.dropna(subset= features, inplace=True)
    print(f"nans removed, removed {len(df) - len(df.dropna())}, removed {Decimal(100 * (len(df) - len(df.dropna()))/len(df)).quantize(Decimal('1.0'))}%")
    print(f"Remaining data has mean Rrc/P ratio: {Decimal(np.nanmean(df['Rrc/P'])).quantize(Decimal('1.00'))}")

    train_params = ['slope_perc' , 'lat', 'lon', 'Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']
    train_params = features

    # aus_X = pd.read_csv(aus_file)[train_params]
    random.seed(seed)
    random_num = random.randint(0, 1000)


    if not test_data:
        X = df[train_params]
        y = df[y_var]
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=t_size, random_state=random_num)

    else:
        train_data_file = 'train_data.csv'

        train_data = pd.read_csv(train_data_file)
        Xtrain = train_data[train_params]
        ytrain = train_data[y_var]

    rf = RandomForestRegressor(n_estimators=trees, max_depth=max_splits, random_state=random_num, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, oob_score=True)
    rf.fit(Xtrain, ytrain)

    print(f'Training Score: {rf.score(Xtrain, ytrain):.3f}')
    print(f'Out of bag score (R2 Score): {rf.oob_score_:.3f}')

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

    #     print('Starting predictions...')
    #     y_pred_aus = pd.DataFrame({'lat': pd.read_csv(aus_file).iloc[:,0], 'lon': pd.read_csv(aus_file).iloc[:,1], y_predict: rf.predict(aus_X)})
    #     print('Finished prediction... writing values')

    #     y_pred_valid.to_csv(f'model_validation_predictions_errors_50_{trees}trees_mf{max_features}_{k_num}fold_out.csv', index=False)
    #     y_pred_aus.to_csv(f'model_predictions_aus_{trees}trees_mf{max_features}_{k_num}fold_out.csv', index=False)
    print(f'Model took: {(datetime.now() - start_time).total_seconds()/60:.2f} minutes to run')

    # # Save the trained model to a file if using test data
    # if test_data:
    #     model_dir = os.path.join(os.path.dirname(__file__), '..', 'Trained_models')
    #     if not os.path.exists(model_dir):
    #         os.makedirs(model_dir)
    #     model_filename = os.path.join(model_dir, f'rf_model_{trees}trees_mf{max_features}.pkl')
    #     joblib.dump(rf, model_filename)
    #     print(f'Model saved to {model_filename}')

if __name__ == "__main__":
    run_rf_model(test_data=True)
