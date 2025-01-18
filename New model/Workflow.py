import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from datetime import datetime
from decimal import Decimal
import random
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


class Workflow():
    def __init__(self, k_num=10, y_var='Recharge RC 50% mm/y', y_predict='R50', aus_file='Australia_grid_0p05_data.csv', seed=42, test_data=False):
        self.DataLocation = os.path.join(os.path.dirname(__file__), 'data')
        self.trainparams = ['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']
        self.aus_file = aus_file
        self.aus_X = pd.read_csv(os.path.join(self.DataLocation, aus_file))[self.trainparams]
        self.seed = seed
        random.seed(self.seed)
        self.random_num = random.randint(0, 1000)
        self.k_num = k_num
        self.y_var = y_var
        self.y_predict = y_predict
        self.test_data = test_data
        self.load_data(self.DataLocation)

    def load_data(self, path):
        self.df = pd.read_csv(os.path.join(path, "dat07_u.csv"), low_memory=False).sample(frac=1, random_state=self.seed)

        if not self.test_data:
            X = self.df[self.trainparams]
            y = self.df[self.y_var]
            t_size = 0.3
            self.Xtrain, self.Xvalid, self.ytrain, self.yvalid = train_test_split(X, y, test_size=t_size, random_state=self.random_num)

        else:
            train_data_file = 'train_data.csv'
            train_data = pd.read_csv(os.path.join(path, train_data_file))
            self.Xtrain = train_data[self.trainparams]
            self.ytrain = train_data[self.y_var]

            validation_data_file = 'test_data.csv'
            validation_data = pd.read_csv(os.path.join(path, validation_data_file))
            self.Xvalid = validation_data[self.trainparams]
            self.yvalid = validation_data[self.y_var]

    def RF_train(self, n_estimators=50, max_depth=40, max_features='log2', min_samples_leaf=8, min_samples_split=8, bootstrap=False, oob_score=False):
        print("Training Random Forest")
        self.rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=self.random_num, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, bootstrap=bootstrap, oob_score=oob_score)
        self.rf.fit(self.Xtrain, self.ytrain)
        print(f'Training Score Random Forest: {self.rf.score(self.Xtrain, self.ytrain):.3f}')

    def RF_cross_validdation(self):
        if not hasattr(self, 'rf'):
            self.RF_train()

        print("Random Forest Cross Validation")
        scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}
        cv_results = cross_validate(self.rf, self.Xtrain, self.ytrain, cv=self.k_num, scoring=scoring, n_jobs=-1)
        print(f'k={self.k_num}')
        print(f'R2 Score: {np.mean(cv_results["test_r2"]):.3f}')
        print(f'RMSE: {-np.mean(cv_results["test_rmse"]):.1f}')
        print(f'MAE: {-np.mean(cv_results["test_mae"]):.1f}')

    def RF_predictions(self, path=os.path.join(os.path.dirname(__file__), 'data')):
        if not hasattr(self, 'rf'):
            self.RF_train()

        print('Starting Random Forest predictions')
        self.rf_y_pred_aus = pd.DataFrame({'lat': pd.read_csv(os.path.join(path, self.aus_file)).iloc[:,0], 'lon': pd.read_csv(os.path.join(path, self.aus_file)).iloc[:,1], self.y_predict: self.rf.predict(self.aus_X)})
        print('Finished prediction... writing values')

        rf_ypredv = self.rf.predict(self.Xvalid)
        self.rf_y_pred_valid = pd.DataFrame({'rf_y_predict': rf_ypredv, 'rf_y_validation': self.yvalid})
        self.rf_y_pred_valid['Residual (predicted R - CMB R)'] = self.rf_y_pred_valid['rf_y_predict'] - self.rf_y_pred_valid['rf_y_validation']
        self.rf_y_pred_valid['Residual (%)'] = ((self.rf_y_pred_valid['rf_y_predict'] - self.rf_y_pred_valid['rf_y_validation']) / self.rf_y_pred_valid['rf_y_validation']) * 100

        self.rf_y_pred_valid.to_csv(f'model_validation_predictions_errors_RF_{datetime.now().date()}.csv', index=False)
        self.rf_y_pred_aus.to_csv(f'model_predictions_aus_RF_{datetime.now().date()}.csv', index=False)

    def XGB_train(self, n_estimators=350, max_depth=12, learning_rate=0.01, min_child_weight=5, subsample=0.7, colsample_bytree=0.6, gamma=0.2, reg_alpha=1, reg_lambda=2):
        print("Training XGBoost")
        self.xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=self.random_num)
        self.xgb.fit(self.Xtrain, self.ytrain)
        print(f'Training Score XGBoost: {self.xgb.score(self.Xtrain, self.ytrain):.3f}')

    def XGB_cross_validation(self):
        if not hasattr(self, 'xgb'):
            self.XGB_train()

        print("XGBoost Cross Validation")
        scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}
        cv_results = cross_validate(self.xgb, self.Xtrain, self.ytrain, cv=self.k_num, scoring=scoring, n_jobs=-1)
        print(f'k={self.k_num}')
        print(f'R2 Score: {np.mean(cv_results["test_r2"]):.3f}')
        print(f'RMSE: {-np.mean(cv_results["test_rmse"]):.1f}')
        print(f'MAE: {-np.mean(cv_results["test_mae"]):.1f}')

    def XGB_predictions(self, path=os.path.join(os.path.dirname(__file__), 'data')):
        if not hasattr(self, 'xgb'):
            self.XGB_train()

        print('Starting XGBoost predictions')
        self.xgb_y_pred_aus = pd.DataFrame({'lat': pd.read_csv(os.path.join(path, self.aus_file)).iloc[:,0], 'lon': pd.read_csv(os.path.join(path, self.aus_file)).iloc[:,1], self.y_predict: self.xgb.predict(self.aus_X)})
        print('Finished prediction... writing values')

        xgb_ypredv = self.xgb.predict(self.Xvalid)
        self.xgb_y_pred_valid = pd.DataFrame({'xgb_y_predict': xgb_ypredv, 'xgb_y_validation': self.yvalid})
        self.xgb_y_pred_valid['Residual (predicted R - CMB R)'] = self.xgb_y_pred_valid['xgb_y_predict'] - self.xgb_y_pred_valid['xgb_y_validation']
        self.xgb_y_pred_valid['Residual (%)'] = ((self.xgb_y_pred_valid['xgb_y_predict'] - self.xgb_y_pred_valid['xgb_y_validation']) / self.xgb_y_pred_valid['xgb_y_validation']) * 100

        self.xgb_y_pred_valid.to_csv(f'model_validation_predictions_errors_XGB_{datetime.now().date()}.csv', index=False)
        self.xgb_y_pred_aus.to_csv(f'model_predictions_aus_XGB_{datetime.now().date()}.csv', index=False)
 
    def validate_models(self, model='all'):
        if model == 'rf':
            if not hasattr(self, 'rf'):
                self.RF_train()
            predictions = self.rf.predict(self.Xvalid)
            r2 = r2_score(self.yvalid, predictions)
            print(f'R2 Score for RF model: {r2:.3f}')
        elif model == 'xgb':
            if not hasattr(self, 'xgb'):
                self.XGB_train()
            predictions = self.xgb.predict(self.Xvalid)
            r2 = r2_score(self.yvalid, predictions)
            print(f'R2 Score for XGB model: {r2:.3f}')
        elif model == 'all':
            if not hasattr(self, 'rf'):
                self.RF_train()
            rf_predictions = self.rf.predict(self.Xvalid)
            rf_r2 = r2_score(self.yvalid, rf_predictions)
            print(f'R2 Score for RF model: {rf_r2:.3f}')

            if not hasattr(self, 'xgb'):
                self.XGB_train()
            xgb_predictions = self.xgb.predict(self.Xvalid)
            xgb_r2 = r2_score(self.yvalid, xgb_predictions)
            print(f'R2 Score for XGB model: {xgb_r2:.3f}')
        else:
            raise ValueError("Model should be 'rf', 'xgb', or 'all'")

    def plot_model_predictions(self, model='rf'):
        if model == 'rf' and not hasattr(self, 'rf_y_pred_aus'):
            self.RF_predictions()
        elif model == 'xgb' and not hasattr(self, 'xgb_y_pred_aus'):
            self.XGB_predictions()

        if model == 'rf':
            data = self.rf_y_pred_aus
            title = 'Random Forest Predictions'
        elif model == 'xgb':
            data = self.xgb_y_pred_aus
            title = 'XGBoost Predictions'

        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(data['lon'], data['lat'], s=0.1, c=data[self.y_predict], cmap='Blues')#, vmax=1000)
        fig.colorbar(sc, ax=ax, label='Recharge rate (mm/yr)')
        ax.set(xlabel=r'Longitude ($\degree$E)', ylabel='Latitude ($\degree$N)', aspect='equal')
        plt.title(title)
        plt.show()

    def plot_parameters(self, plot_type='training'):
        if plot_type == 'training':
            data = self.df
            params = self.trainparams
            title = 'Training Parameters'

        elif plot_type == 'prediction':
            select = self.trainparams + ["lat", "lon"]
            data = pd.read_csv(os.path.join(self.DataLocation, self.aus_file))[select]
            params = self.trainparams
            title = 'Prediction parameters'

        fig, axes = plt.subplots(nrows=4, ncols=len(params)//4 + (len(params) % 4 > 0), figsize=(20, 15))
        axes = axes.flatten()
        for i, param in enumerate(params):
            ax = axes[i]
            sc = ax.scatter(data['lon'], data['lat'], s=0.1, c=data[param], cmap='viridis', alpha=0.5, label=param)
            fig.colorbar(sc, ax=ax, label=param)
            ax.set_xlabel('Longitude ($\degree$E)')
            ax.set_ylabel('Latitude ($\degree$N)')
            ax.set_aspect('equal', 'box')
            ax.legend(loc='lower left')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    workflow = Workflow(test_data=True)
    workflow.RF_train(n_estimators=500, max_depth=25, max_features='log2', min_samples_leaf=3, oob_score=True, bootstrap=True)
    workflow.validate_models()

