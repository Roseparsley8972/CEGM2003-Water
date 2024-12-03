import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from datetime import datetime
from decimal import Decimal
import random
from sklearn.model_selection import cross_validate


def run_rf_model(t_size=0.3, trees=250, max_splits=18, max_features=0.33, min_samples_leaf=8, min_samples_split=2, k_num=10, y_var='Recharge RC 50% mm/y', y_predict='R50', aus_file='Australia_grid_0p05_data.csv', seed=42):
    start_time = datetime.now()

    # Load libraries
    DataLocation = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.chdir(DataLocation)

    y_pred_valid = pd.DataFrame()
    y_pred_aus = pd.DataFrame()

    print('Recharge 50%')
    df = pd.read_csv('dat07_u.csv', low_memory=False)
    df2 = df.sample(frac=1, random_state=seed)
    df2.dropna(subset=['Rain mm/y', 'koppen_geiger', 'PET mm/y', 'distance_to_coast_km', 'Aridity', 'elevation_mahd', 'wtd_mbgs', 'regolith_depth_mbgs', 'slope_perc', 'clay_perc', 'silt_perc', 'sand_perc', 'soil_class', 'geology', 'ndvi_avg', 'vegex_cat', 'rainfall_seasonality'], inplace=True)
    print("nans removed, removed n=" + str(len(df) - len(df2)) + ", removed " + str(Decimal(100 * (len(df) - len(df2))/len(df)).quantize(Decimal("1.0"))) + "%")
    print("Remaining data has mean Rrc/P ratio:" + str(Decimal(np.nanmean(df2['Rrc/P'])).quantize(Decimal("1.00"))))

    X = df2[['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']]
    y = df2[y_var]
    aus_dat = pd.read_csv(aus_file)
    aus_X = aus_dat[['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']]

    random.seed(seed)
    random_num = random.randint(0, 1000)

    Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=t_size, random_state=random_num)

    rf = RandomForestRegressor(n_estimators=trees, max_depth=max_splits, random_state=random_num, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, oob_score=True)
    rf.fit(Xtrain, ytrain)

    training_score = rf.score(Xtrain, ytrain)
    print(f'Training Score: {training_score:.3f}')

    oob_r2 = rf.oob_score_
    print(f'Out of bag score (R2 Score): {oob_r2:.3f}')

    ypredv = rf.predict(Xvalid)
    y_pred_valid = pd.DataFrame({'y_predict': ypredv, 'y_validation': yvalid})
    y_pred_valid['Residual (predicted R - CMB R)'] = y_pred_valid['y_predict'] - y_pred_valid['y_validation']
    y_pred_valid['Residual (%)'] = ((y_pred_valid['y_predict'] - y_pred_valid['y_validation']) / y_pred_valid['y_validation']) * 100
    r2 = r2_score(yvalid, ypredv)
    rmse = mean_squared_error(yvalid, ypredv, squared=False)
    mae = mean_absolute_error(yvalid, ypredv)
    print(f'R2 Score: {r2:.3f}')
    print(f'RMSE: {rmse:.3f}')
    print(f'MAE: {mae:.3f}')

    scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}
    cv_results = cross_validate(rf, Xtrain, ytrain, cv=k_num, scoring=scoring, n_jobs=-1)

    r2_cv = np.mean(cv_results['test_r2'])
    mae_cv = np.mean(-cv_results['test_mae'])
    rmse_cv = np.mean(-cv_results['test_rmse'])
    print(f'k={k_num}')
    print(f'R2 Score: {r2_cv:.3f}')
    print(f'RMSE: {rmse_cv:.1f}')
    print(f'MAE: {mae_cv:.1f}')

    print('Starting predictions...')
    ypred = rf.predict(aus_X)
    y_pred_aus['lat'] = aus_dat.iloc[:,0]
    y_pred_aus['lon'] = aus_dat.iloc[:,1]
    y_pred_aus[y_predict] = ypred
    print('Finished prediction... writing values')

    y_pred_valid.to_csv(f'model_validation_predictions_errors_50_{trees}trees_mf{max_features}_{k_num}fold_out.csv', index=False)

    print('Saving file')
    y_pred_aus.to_csv(f'model_predictions_aus_{trees}trees_mf{max_features}_{k_num}fold_out.csv', index=False)
    print(f'Model took: {(datetime.now() - start_time).total_seconds()/60:.2f} minutes to run')

if __name__ == "__main__":
    run_rf_model()
