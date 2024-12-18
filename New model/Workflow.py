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


class Workflow():
    def __init__(self, k_num=10, y_var='Recharge RC 50% mm/y', y_predict='R50', aus_file='Australia_grid_0p05_data.csv', seed=42, test_data=False):
        DataLocation = os.path.join(os.path.dirname(__file__), 'data')
        self.trainparams = ['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']
        self.aus_X = pd.read_csv(os.path.join(DataLocation, aus_file))[self.trainparams]
        self.seed = seed
        random.seed(self.seed)
        self.random_num = random.randint(0, 1000)
        self.k_num = k_num
        self.y_var = y_var
        self.y_predict = y_predict
        self.test_data = test_data
        self.load_data(DataLocation)

        print(self.Xtrain)

    def load_data(self, path):
        if not self.test_data:
            df = pd.read_csv(os.path.join(path, "dat07_u.csv"), low_memory=False).sample(frac=1, random_state=self.seed)
            X = df[self.trainparams]
            y = df[self.y_var]
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
        self.rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=self.random_num, max_features='log2', min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, bootstrap=bootstrap, oob_score=oob_score)
        rf.fit(Xtrain, ytrain)

    print(f'Training Score: {rf.score(Xtrain, ytrain):.3f}')


if __name__ == "__main__":
    Workflow(test_data=True)
