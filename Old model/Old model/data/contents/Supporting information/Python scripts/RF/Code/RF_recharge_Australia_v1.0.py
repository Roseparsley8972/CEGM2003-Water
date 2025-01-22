# -*- coding: utf-8 -*-
"""
Created on 20 September 2023 at 16:45
RF_recharge_Australia_v1.0.py trains random forest regression models using CMB
generated recharge estimates (R5, R50, R95) and eight most important variables.
The random forest regression models are validated using out-of-bag R-squared
score, an external validation R-squared, and a 10-fold cross-validation
R-squared score. After validation of model performance, a 0.05 degree latitude/
longitude grid covering Australia along with corresponding values from the
eight most important variables are used to produce recharge estimates (R5, R50
and R95) for each point in the grid. The output can be exported to QGIS to
create gridded map outputs (e.g., .tif rasters).

@author: slee and dirvine
"""
#%% load libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from decimal import Decimal
import random
from datetime import datetime
#=============================================================================
#%% Data Preparation
InputData = os.path.join(os.path.dirname(__file__), '..', 'InputData')
OutputFiles = os.path.join(os.path.dirname(__file__), '..', 'OutputFiles')
# User settings
usedatasplit = 'yes' # Yes to Split dataset into training and testing sets
t_size = 0.3 #specify 30% data for testing/validation purposes
trees = [250]
max_splits = 18
max_features = 0.33
max_leaf_nodes = None
min_samples_leaf = 8
min_samples_split = 2
k_num = 10 # previously tried 10 for k-fold validation
x_group=['8par_grp6']
y_group = ['5', '50', '95']
y_group_labs = ['Recharge 5th percentile', 'Median Recharge', 'Recharge 95th percentile']
y_var = ['Recharge RC 5% mm/y', 'Recharge RC 50% mm/y', 'Recharge RC 95% mm/y']
y_predict = ['R5', 'R50', 'R95']
y_pred_valid = pd.DataFrame()
y_pred_aus = pd.DataFrame()
random_nums = [] # create empty list to write random numbers
aus_file = 'Australia_grid_0p05_data_new.csv'

#%%
#Start loop to go through mean, lower and upper
for j in range(len(y_group)):
    print('Recharge ' + str(y_group[j]))
    os.chdir(InputData)
    df= pd.read_csv('dat07_u.csv')
    df2 = df.sample(frac=1, random_state=42) # randomly shuffle rows of dataset
    df2.dropna(subset=['Rain mm/y', 'koppen_geiger', 'PET mm/y', 'distance_to_coast_km', 'Aridity', 'elevation_mahd', 'wtd_mbgs', 'regolith_depth_mbgs', 'slope_perc', 'clay_perc', 'silt_perc', 'sand_perc', 'soil_class', 'geology', 'ndvi_avg', 'vegex_cat', 'rainfall_seasonality'], inplace=True) # remove blank cells from dataframe variables. Remove variable if needed ('distance_to_coast_km')
    print("nans removed, removed n=" + str(len(df) - len(df2)) + ", removed " + str(Decimal(100 * (len(df) - len(df2))/len(df)).quantize(Decimal("1.0"))) + "%")
    print("Remaining data has mean Rrc/P ratio:" + str(Decimal(np.nanmean(df2['Rrc/P'])).quantize(Decimal("1.00"))))
    # Separate the dependent and independent variables
    X = df2.drop(columns=y_var[j])
    y = df2[y_var[j]]
    os.chdir(InputData)
    aus_dat= pd.read_csv(aus_file)
    aus_X = aus_dat[['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']]
    # Previous code testing
    for k in range(len(x_group)):
        print('X group ' + str(x_group[k]))
        if k == 0:
            X = df2[['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']] #8par_grp6
        features = list(X.columns) # names for plotting later
        model_test_df = pd.DataFrame(columns=['trees', 'oob_r2_score', 'valid_r2_score', 'valid_rmse', 'valid_mae', f'{k_num}-fold_r2_score', f'{k_num}-fold_rmse', f'{k_num}-fold_mae'])
        fimp_df = pd.DataFrame(columns=features) # df to write/append feature importance scores
        for i in range(len(trees)):
            print(datetime.now())
            print('Trees: ' + str(trees[i]))
            seed = 42 # Set seed to a fixed value
            random.seed(seed)
            random_num = (random.randint(0, 1000))
            random_nums.append(random_num)  
            if 'y' in usedatasplit:
                Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size = t_size, random_state = random_num) #split dataset into training (70%) and validation or testing (30%) but ignore testing/validation data
                # Model Random Forest Regression
                rf = RandomForestRegressor(n_estimators=trees[i], max_depth=max_splits,
                                           random_state=random_num, max_features=max_features,
                                           max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                           min_samples_split=min_samples_split,
                                           oob_score=True) # n estimators is the number of trees in each forest, max depth is the maximum decision splits for each tree, seed ensures randomisation is the same each time for reproducibility
                rf.fit(Xtrain, ytrain) # Train the data on just training data
            
                # Training score
                training_score = rf.score(Xtrain, ytrain)
                print(f'Training Score: {training_score:.3f}')
                
                # Internal validation using out of bag (OOB) samples (from training data)
                oob_r2 = rf.oob_score_ # This is R2 value of oob predictions
                print('Internal validation using out of bag (training data)')
                print(f'Out of bag score (R2 Score): {oob_r2:.3f}')
                
                # External validation using 30% samples which are separate to the training data
                ypredv = rf.predict(Xvalid) # extract predictions on the testing set
                X_validation = Xvalid.to_numpy()
                y_pred_valid = pd.DataFrame(ypredv, columns = ['y_predict']) # Save y predictions to a column of a dataframe named y_pred_valid
                y_pred_valid['y_validation'] = yvalid.array
                y_pred_valid['Residual (predicted R - CMB R)'] = y_pred_valid['y_predict'] - y_pred_valid['y_validation']
                y_pred_valid['Residual (%)'] = ((y_pred_valid['y_predict'] - y_pred_valid['y_validation']) / y_pred_valid['y_validation']) * 100
                r2 = r2_score(yvalid, ypredv)
                rmse = mean_squared_error(yvalid, ypredv, squared=False) #If squared is True returns MSE value, if False returns RMSE value
                mae = mean_absolute_error(yvalid, ypredv)
                print('External validation using validation data')
                print(f'R2 Score: {r2:.3f}')
                print(f'RMSE: {rmse:.3f}')
                print(f'MAE: {mae:.3f}')
                
                # K-fold cross-validation testing
                scoring = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
                cv_results = cross_validate(rf, Xtrain, ytrain, cv=k_num, scoring=scoring) #10-fold cross validation using 70% of data
                # Extract the mean of each metric from cross-validation results
                r2_cv = np.mean(cv_results['test_r2'])
                mae_cv = np.mean(-cv_results['test_neg_mean_absolute_error'])
                rmse_cv = np.mean(-cv_results['test_neg_root_mean_squared_error'])
                print('Cross validation using k-fold test')
                print(f'k={k_num}')
                print(f'R2 Score: {r2_cv:.3f}')
                print(f'RMSE: {rmse_cv:.1f}')
                print(f'MAE: {mae_cv:.1f}')
                
                # Predict for all 0.05 x 0.05 degree pixels in Australia
                print('Starting predictions...')
                print(datetime.now())
                ypred = rf.predict(aus_X) # extract predictions on the testing set
                print('Finished prediction... writing values')
                print(datetime.now())
                X_aus = aus_X.to_numpy()
                if j == 0:
                    y_pred_aus['lat'] = aus_dat.iloc[:,0]
                    y_pred_aus['lon'] = aus_dat.iloc[:,1]
                    y_pred_aus[y_predict[j]] = ypred 
                else:
                    y_pred_aus[y_predict[j]] = ypred # Save y predictions to a column of a dataframe named y_pred_valid
                print(datetime.now())
            # Output stats
            os.chdir(OutputFiles)
            y_pred_valid.to_csv('model_validation_predictions_errors_' + str(y_group[j]) + '_' + str(x_group[k]) + '_' + str(trees[i]) + 'trees_mf' + str(max_features) + f'_{k_num}fold_out.csv', index=False)
# Output stats
print('Saving file')
os.chdir(OutputFiles)
y_pred_aus.to_csv('model_predictions_aus_' + str(x_group[k]) + '_' + str(trees[i]) + 'trees_mf' + str(max_features) + f'_{k_num}fold_out.csv', index=False)
print(datetime.now())