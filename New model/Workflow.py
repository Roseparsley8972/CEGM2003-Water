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
from sklearn.linear_model import Lasso
from matplotlib.colors import LogNorm



class Workflow():
    """
    A class to represent a machine learning workflow for predicting recharge rates using Random Forest and XGBoost models.
    Attributes
    ----------
    DataLocation : str
        Path to the data directory.
    trainparams : list
        List of training parameters.
    aus_file : str
        Filename of the Australian grid data.
    aus_X : DataFrame
        DataFrame containing the Australian grid data.
    seed : int
        Random seed for reproducibility.
    random_num : int
        Random number generated using the seed.
    k_num : int
        Number of folds for cross-validation.
    y_var : str
        Target variable for training.
    y_predict : str
        Name of the prediction column.
    df : DataFrame
        DataFrame containing the loaded data.
    Xtrain : DataFrame
        Training features.
    Xvalid : DataFrame
        Validation features.
    ytrain : Series
        Training target.
    yvalid : Series
        Validation target.
    rf : RandomForestRegressor
        Random Forest model.
    xgb : XGBRegressor
        XGBoost model.
    rf_y_pred_aus : DataFrame
        DataFrame containing Random Forest predictions for Australia.
    rf_y_pred_valid : DataFrame
        DataFrame containing Random Forest validation predictions and errors.
    xgb_y_pred_aus : DataFrame
        DataFrame containing XGBoost predictions for Australia.
    xgb_y_pred_valid : DataFrame
        DataFrame containing XGBoost validation predictions and errors.
    Methods
    -------
    __init__(k_num=10, y_var='Recharge RC 50% mm/y', y_predict='R50', aus_file='Australia_grid_0p05_data.csv', seed=42):
        Initializes the Workflow with the given parameters.
    load_data(path):
        Loads the data from the specified path.
    RF_train(n_estimators=50, max_depth=40, max_features='log2', min_samples_leaf=8, min_samples_split=8, bootstrap=False, oob_score=False):
        Trains the Random Forest model with the specified parameters.
    RF_cross_validdation():
        Performs cross-validation for the Random Forest model.
    RF_predictions(path=os.path.join(os.path.dirname(__file__), 'data')):
        Generates predictions using the Random Forest model and saves the results to CSV files.
    XGB_train(n_estimators=350, max_depth=12, learning_rate=0.01, min_child_weight=5, subsample=0.7, colsample_bytree=0.6, gamma=0.2, reg_alpha=1, reg_lambda=2):
        Trains the XGBoost model with the specified parameters.
    XGB_cross_validation():
        Performs cross-validation for the XGBoost model.
    XGB_predictions(path=os.path.join(os.path.dirname(__file__), 'data')):
        Generates predictions using the XGBoost model and saves the results to CSV files.
    validate_models(model='all'):
        Validates the specified model(s) and prints the R2 score.
    plot_model_predictions(model='rf'):
        Plots the predictions of the specified model.
    plot_parameters(plot_type='training'):
        Plots the training or prediction parameters.
    """
    def __init__(self, k_num=10, y_var='Recharge RC 50% mm/y', y_predict='R50', aus_file='Australia_grid_0p05_data.csv', seed=42):
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
        self.load_data(self.DataLocation)

    def load_data(self, path=os.path.join(os.path.dirname(__file__), 'data')):
        """
        Loads data from the specified path and splits it into training, validation, and test sets.
        Parameters:
        path (str): The directory path where the data files are located.
        Attributes:
        self.df (DataFrame): The loaded and shuffled DataFrame.
        self.Xtrain (DataFrame): The training features.
        self.Xvalid (DataFrame): The validation features.
        self.ytrain (Series): The training target variable.
        self.yvalid (Series): The validation target variable.
        self.Xtest (DataFrame): The test features.
        self.ytest (Series): The test target variable.
        """

        self.df = pd.read_csv(os.path.join(path, "dat07_u.csv"), low_memory=False).sample(frac=1, random_state=self.seed)

        train_data_file = 'train_data.csv'
        train_data = pd.read_csv(os.path.join(path, train_data_file))
        self.Xtrain = train_data[self.trainparams]
        self.ytrain = train_data[self.y_var]

        validation_data_file = 'validation_data.csv'
        validation_data = pd.read_csv(os.path.join(path, validation_data_file))
        self.Xvalid = validation_data[self.trainparams]
        self.yvalid = validation_data[self.y_var]

        test_data_file = 'test_data.csv'
        test_data = pd.read_csv(os.path.join(path, test_data_file))
        self.Xtest = test_data[self.trainparams]
        self.ytest = test_data[self.y_var]

        return self.Xtrain, self.ytrain, self.Xvalid, self.yvalid, self.Xtest, self.ytest

    def RF_train(self, n_estimators=90, max_depth=50, max_features='log2', min_samples_leaf=6, min_samples_split=4, 
                bootstrap=False, oob_score=False, old_model=False):
        """
        Trains a Random Forest Regressor model using the provided training data.
        
        Parameters:
        n_estimators (int): The number of trees in the forest. Default is 90.
        max_depth (int): The maximum depth of the tree. Default is 50.
        max_features (str): The number of features to consider when looking for the best split. Default is 'log2'.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node. Default is 6.
        min_samples_split (int): The minimum number of samples required to split an internal node. Default is 4.
        bootstrap (bool): Whether bootstrap samples are used when building trees. Default is False.
        oob_score (bool): Whether to use out-of-bag samples to estimate the generalization accuracy. Default is False.
        Returns:
        None
        
        Prints:
        Training progress and the training score of the Random Forest model.
        """

        if not old_model:
            print("Training Random Forest")
            
            self.rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=self.random_num, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, bootstrap=bootstrap, oob_score=oob_score)
            self.rf.fit(self.Xtrain, self.ytrain)
            print(f'Training Score Random Forest: {self.rf.score(self.Xtrain, self.ytrain):.3f}')

            if oob_score:
                 print(f'OOB Score Random Forest: {self.rf.oob_score_:.3f}')
        if old_model:
            trees = 250
            max_splits = 18
            max_features = 0.33
            max_leaf_nodes = None
            min_samples_leaf = 8
            min_samples_split = 2

            self.rf_old = RandomForestRegressor(n_estimators=trees, max_depth=max_splits,
                                           random_state=self.seed, max_features=max_features,
                                           max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                           min_samples_split=min_samples_split,
                                           oob_score=True)
            
            self.rf_old.fit(self.Xtrain, self.ytrain)
            print(f'Training Score Random Forest: {self.rf_old.score(self.Xtrain, self.ytrain):.3f}')

    def RF_cross_validation(self):
        """
        Perform cross-validation on the Random Forest model and print the results.
        This method checks if the Random Forest model has been trained. If not, it calls the RF_train method to train the model.
        It then performs cross-validation using the specified number of folds (k_num) and calculates the R2 score, 
        mean absolute error (MAE), and root mean squared error (RMSE) for each fold. The results are printed to the console.
        Attributes:
            rf (RandomForestRegressor): The Random Forest model.
            Xtrain (pd.DataFrame or np.ndarray): The training data features.
            ytrain (pd.Series or np.ndarray): The training data target values.
            k_num (int): The number of folds for cross-validation.
        Prints:
            str: A message indicating that cross-validation is being performed.
            str: The number of folds used in cross-validation.
            str: The mean R2 score across all folds.
            str: The mean RMSE across all folds.
            str: The mean MAE across all folds.
        """

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
        """
        Generate Random Forest predictions for Australian data and validation data, and save the results to CSV files.
        Parameters:
        path (str): The directory path where the input data files are located. Defaults to a 'data' directory in the same location as the script.
        Returns:
        None
        This method performs the following steps:
        1. Checks if the Random Forest model ('rf') is trained. If not, it calls the RF_train() method to train the model.
        2. Reads the latitude and longitude data from the Australian data file and makes predictions using the trained Random Forest model.
        3. Creates a DataFrame with the predictions and saves it to a CSV file named 'model_predictions_aus_RF_<current_date>.csv'.
        4. Makes predictions on the validation dataset and calculates residuals and residual percentages.
        5. Creates a DataFrame with the validation predictions, residuals, and residual percentages, and saves it to a CSV file named 'model_validation_predictions_errors_RF_<current_date>.csv'.
        """
        
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

    def XGB_train(self, n_estimators=400, max_depth=12, learning_rate=0.01, min_child_weight=5, subsample=0.6,
                colsample_bytree=0.6, gamma=0.2, reg_alpha=0, reg_lambda=2.5):
        """
        Trains an XGBoost regressor model with the given hyperparameters.
        Parameters:
        n_estimators (int): Number of boosting rounds. Default is 400.
        max_depth (int): Maximum depth of a tree. Default is 12.
        learning_rate (float): Boosting learning rate. Default is 0.01.
        min_child_weight (int): Minimum sum of instance weight (hessian) needed in a child. Default is 5.
        subsample (float): Subsample ratio of the training instances. Default is 0.6.
        colsample_bytree (float): Subsample ratio of columns when constructing each tree. Default is 0.6.
        gamma (float): Minimum loss reduction required to make a further partition on a leaf node of the tree. Default is 0.2.
        reg_alpha (float): L1 regularization term on weights. Default is 0.
        reg_lambda (float): L2 regularization term on weights. Default is 2.5.
        Returns:
        None
        """

        print("Training XGBoost")
        self.xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=self.random_num)
        self.xgb.fit(self.Xtrain, self.ytrain)
        print(f'Training Score XGBoost: {self.xgb.score(self.Xtrain, self.ytrain):.3f}')

    def XGB_cross_validation(self):
        """
        Perform cross-validation using the XGBoost model.
        This method checks if the XGBoost model has been trained. If not, it trains the model first.
        Then, it performs cross-validation on the training data using the specified number of folds (k_num).
        The method prints the cross-validation results, including the R2 score, RMSE, and MAE.
        Attributes:
            xgb: The trained XGBoost model.
            Xtrain: The training data features.
            ytrain: The training data target values.
            k_num: The number of folds for cross-validation.
        Prints:
            The number of folds used in cross-validation.
            The mean R2 score from cross-validation.
            The mean RMSE from cross-validation.
            The mean MAE from cross-validation.
        """

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
        """
        Generates predictions using the XGBoost model and saves the results to CSV files.
        Parameters:
        path (str): The directory path where the input data files are located. Defaults to a 'data' directory 
                in the same location as the script.
        This method performs the following steps:
        1. Checks if the XGBoost model is trained. If not, it calls the `XGB_train` method to train the model.
        2. Reads the latitude and longitude data from the Australian dataset file.
        3. Uses the trained XGBoost model to predict values for the Australian dataset and stores the results in a DataFrame.
        4. Uses the trained XGBoost model to predict values for the validation dataset and stores the results in a DataFrame.
        5. Calculates the residuals and percentage residuals for the validation predictions.
        6. Saves the validation predictions and residuals to a CSV file.
        7. Saves the Australian dataset predictions to a CSV file.
        The output CSV files are named with the current date to ensure uniqueness.
        """

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
 
    def Lasso_train(self, alpha=0.01, max_iter=1000, tol=0.001):
        """
        1. Initializes a Lasso regression model with the given alpha and random seed.
        2. Fits the model to the training data (`Xtrain` and `ytrain`).
        3. Prints the training score (R2 score) of the model on the training dataset.
        action: 
        Trains a Lasso regression model with the given hyperparameter.
        Parameters:
        alpha (float): The regularization strength. Default is 0.1.
                    A smaller value indicates weaker regularization.
        returns: 
        training score lasso
            
        """
        print("Training Lasso")
        self.lasso = Lasso(alpha=alpha, max_iter=max_iter, tol=tol, random_state=self.random_num)
        self.lasso.fit(self.Xtrain, self.ytrain)
        print(f'Training Score Lasso: {self.lasso.score(self.Xtrain, self.ytrain):.3f}')

    def Lasso_cross_validation(self):

        """
        Performs cross-validation using the Lasso regression model.
        This method:
        1. Checks if the Lasso model has been trained. If not, it trains the model first.
        2. Performs k-fold cross-validation on the training dataset.
        3. Evaluates and prints the mean R2 score, RMSE, and MAE across all folds.
        Parameters:
        None
        Attributes:
            lasso: The trained Lasso regression model.
            Xtrain: The training data features.
            ytrain: The training data target values.
            k_num: The number of folds for cross-validation.
        Prints:
            - Number of folds used in cross-validation.
            - Mean R2 score from cross-validation.
            - Mean RMSE from cross-validation.
            - Mean MAE from cross-validation.
        """
        if not hasattr(self, 'lasso'):
            self.Lasso_train()


        if not hasattr(self, 'lasso'):
            self.Lasso_train()

        print("Lasso Cross Validation")
        scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}
        cv_results = cross_validate(self.lasso, self.Xtrain, self.ytrain, cv=self.k_num, scoring=scoring, n_jobs=-1)
        print(f'k={self.k_num}')
        print(f'R2 Score: {np.mean(cv_results["test_r2"]):.3f}')
        print(f'RMSE: {-np.mean(cv_results["test_rmse"]):.1f}')
        print(f'MAE: {-np.mean(cv_results["test_mae"]):.1f}')

    def Lasso_predictions(self, path=os.path.join(os.path.dirname(__file__), 'data')):
        if not hasattr(self, 'lasso'):
            self.Lasso_train()

        print('Starting Lasso predictions')
        self.lasso_y_pred_aus = pd.DataFrame({'lat': pd.read_csv(os.path.join(path, self.aus_file)).iloc[:,0], 'lon': pd.read_csv(os.path.join(path, self.aus_file)).iloc[:,1], self.y_predict: self.lasso.predict(self.aus_X)})
        print('Finished prediction... writing values')

        lasso_ypredv = self.lasso.predict(self.Xvalid)
        self.lasso_y_pred_valid = pd.DataFrame({'lasso_y_predict': lasso_ypredv, 'lasso_y_validation': self.yvalid})
        self.lasso_y_pred_valid['Residual (predicted R - CMB R)'] = self.lasso_y_pred_valid['lasso_y_predict'] - self.lasso_y_pred_valid['lasso_y_validation']
        self.lasso_y_pred_valid['Residual (%)'] = ((self.lasso_y_pred_valid['lasso_y_predict'] - self.lasso_y_pred_valid['lasso_y_validation']) / self.lasso_y_pred_valid['lasso_y_validation']) * 100

        self.lasso_y_pred_valid.to_csv(f'model_validation_predictions_errors_Lasso_{datetime.now().date()}.csv', index=False)
        self.lasso_y_pred_aus.to_csv(f'model_predictions_aus_lasso_{datetime.now().date()}.csv', index=False)


    def validate_models(self, model='all'):
        """
        Validate the performance of the trained models on the validation dataset.
        Parameters:
        model (str): The model to validate. Options are 'rf' for Random Forest, 'xgb' for XGBoost, 'lasso' for Lasso 
                     or 'all' to validate both models. Default is 'all'.
        Raises:
        ValueError: If the model parameter is not 'rf', 'xgb', 'lasso', or 'all'.
        This method prints the R2 score for the specified model(s) on the validation dataset.
        If the specified model is not trained, it will be trained before validation.
        """

        if model == 'rf':
            if not hasattr(self, 'rf'):
                self.RF_train()
            predictions = self.rf.predict(self.Xvalid)
            r2 = r2_score(self.yvalid, predictions)
            print(f'R2 Score for RF model (Validation data): {r2:.3f}')
        elif model == 'xgb':
            if not hasattr(self, 'xgb'):
                self.XGB_train()
            predictions = self.xgb.predict(self.Xvalid)
            r2 = r2_score(self.yvalid, predictions)
            print(f'R2 Score for XGB model (Validation data): {r2:.3f}')
        elif model == 'lasso':
            if not hasattr(self, 'lasso'):
                self.Lasso_train()
            predictions = self.lasso.predict(self.Xvalid)
            r2 = r2_score(self.yvalid, predictions)
            print(f'R2 Score for Lasso model (Validation data): {r2:.3f}')

        elif model == 'old_rf':
            if not hasattr(self, 'rf_old'):
                self.RF_train(old_model=True)
            predictions = self.rf_old.predict(self.Xvalid)
            r2 = r2_score(self.yvalid, predictions)
            print(f'R2 Score for old RF model (Validation data): {r2:.3f}')   

        elif model == 'all':
            start_time = datetime.now()
            if not hasattr(self, 'rf'):
                self.RF_train()
            rf_predictions = self.rf.predict(self.Xvalid)
            rf_r2 = r2_score(self.yvalid, rf_predictions)
            print(f'R2 Score for RF model (Validation data): {rf_r2:.3f}')
            print(f'Time taken to train RF model: {datetime.now() - start_time}')

            start_time = datetime.now()
            if not hasattr(self, 'xgb'):
                self.XGB_train()
            xgb_predictions = self.xgb.predict(self.Xvalid)
            xgb_r2 = r2_score(self.yvalid, xgb_predictions)
            print(f'R2 Score for XGB model (Validation data): {xgb_r2:.3f}')

            if not hasattr(self, 'lasso'):
                self.Lasso_train()
            predictions = self.lasso.predict(self.Xvalid)
            r2 = r2_score(self.yvalid, predictions)
            print(f'R2 Score for Lasso model (Validation data): {r2:.3f}')
            print(f'Time taken to train XGB model: {datetime.now() - start_time}')

            start_time = datetime.now()
            if not hasattr(self, 'rf_old'):
                self.RF_train(old_model=True)
            predictions = self.rf_old.predict(self.Xvalid)
            r2 = r2_score(self.yvalid, predictions)
            print(f'R2 Score for old RF model (Validation data): {r2:.3f}')
            print(f'Time taken to train old RF model: {datetime.now() - start_time}')

        else:
            raise ValueError("Model should be 'rf', 'xgb', 'lasso, 'old_rd' or 'all'")

    def test_models(self, model='all'):
        """
        Validate the performance of the trained models on the test dataset.
        Parameters:
        model (str): The model to validate. Options are 'rf' for Random Forest, 'xgb' for XGBoost, 'lasso' for Lasso 
                    or 'all' to validate both models. Default is 'all'.
        Raises:
        ValueError: If the model parameter is not 'rf', 'xgb', 'lasso', or 'all'.
        This method prints the R2 score for the specified model(s) on the test dataset.
        If the specified model is not trained, it will be trained before validation.
        """
        if model == 'rf':
            if not hasattr(self, 'rf'):
                self.RF_train()
            predictions = self.rf.predict(self.Xtest)
            r2 = r2_score(self.ytest, predictions)
            print(f'R2 Score for RF model (Test data): {r2:.3f}')
        elif model == 'xgb':
            if not hasattr(self, 'xgb'):
                self.XGB_train()
            predictions = self.xgb.predict(self.Xtest)
            r2 = r2_score(self.ytest, predictions)
            print(f'R2 Score for XGB model (Test data): {r2:.3f}')
        elif model == 'lasso':
            if not hasattr(self, 'lasso'):
                self.Lasso_train()
            predictions = self.lasso.predict(self.Xtest)
            r2 = r2_score(self.ytest, predictions)
            print(f'R2 Score for Lasso model (Test data): {r2:.3f}')
        elif model == 'old_rf':
            if not hasattr(self, 'rf_old'):
                self.RF_train(old_model=True)
            predictions = self.rf_old.predict(self.Xtest)
            r2 = r2_score(self.ytest, predictions)
            print(f'R2 Score for old RF model (Test data): {r2:.3f}')   
        elif model == 'all':
            start_time = datetime.now()
            if not hasattr(self, 'rf'):
                self.RF_train()
            rf_predictions = self.rf.predict(self.Xtest)
            rf_r2 = r2_score(self.ytest, rf_predictions)
            print(f'R2 Score for RF model (Test data): {rf_r2:.3f}')
            print(f'Time taken to train RF model: {datetime.now() - start_time}')
            start_time = datetime.now()
            if not hasattr(self, 'xgb'):
                self.XGB_train()
            xgb_predictions = self.xgb.predict(self.Xtest)
            xgb_r2 = r2_score(self.ytest, xgb_predictions)
            print(f'R2 Score for XGB model (Test data): {xgb_r2:.3f}')
            if not hasattr(self, 'lasso'):
                self.Lasso_train()
            lasso_predictions = self.lasso.predict(self.Xtest)
            lasso_r2 = r2_score(self.ytest, lasso_predictions)
            print(f'R2 Score for Lasso model (Test data): {lasso_r2:.3f}')
            print(f'Time taken to train XGB model: {datetime.now() - start_time}')
            start_time = datetime.now()
            if not hasattr(self, 'rf_old'):
                self.RF_train(old_model=True)
            old_rf_predictions = self.rf_old.predict(self.Xtest)
            old_rf_r2 = r2_score(self.ytest, old_rf_predictions)
            print(f'R2 Score for old RF model (Test data): {old_rf_r2:.3f}')
            print(f'Time taken to train old RF model: {datetime.now() - start_time}')
        else:
            raise ValueError("Model should be 'rf', 'xgb', 'lasso', 'old_rf' or 'all'")

    def plot_model_predictions(self, model='rf'):
        """
        Plots the model predictions on a scatter plot.
        Parameters:
        model (str): The model to use for predictions. Options are 'rf' for Random Forest, 'xgb' for XGBoost and 'lasso' for Lasso. Default is 'rf'.
        This function checks if the predictions for the specified model are already computed. 
        If not, it computes the predictions by calling the appropriate method (RF_predictions, XGB_predictions or Lasso_predictions).
        It then creates a scatter plot of the predictions with longitude and latitude on the x and y axes, respectively.
        The color of the points represents the predicted recharge rate.
        The plot includes:
        - A color bar indicating the recharge rate in mm/yr.
        - Labels for the x and y axes (Longitude and Latitude).
        - A title indicating which model's predictions are being plotted.
        Raises:
        AttributeError: If the specified model's predictions are not available and cannot be computed.
        """

        if model == 'rf' and not hasattr(self, 'rf_y_pred_aus'):
            self.RF_predictions()
        elif model == 'xgb' and not hasattr(self, 'xgb_y_pred_aus'):
            self.XGB_predictions()
        elif model == 'lasso' and not hasattr(self, 'lasso_y_pred_aus'):
            self.Lasso_predictions()
        if model == 'rf':
            data = self.rf_y_pred_aus
            title = 'Random Forest Predictions'
        elif model == 'xgb':
            data = self.xgb_y_pred_aus
            title = 'XGBoost Predictions'
        elif model == 'lasso':
            data = self.lasso_y_pred_aus
            title = 'Lasso Predictions'

        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(data['lon'], data['lat'], s=0.1, c=data[self.y_predict], cmap='Blues')#, vmax=1000)
        fig.colorbar(sc, ax=ax, label='Recharge rate (mm/yr)')
        ax.set(xlabel=r'Longitude ($\degree$E)', ylabel='Latitude ($\degree$N)', aspect='equal')
        plt.title(title)
        plt.show()

    def plot_parameters(self, plot_type='training', plot_rainfall_only=False, plot_recharge_only=False):
        """
        Plots the parameters on a scatter plot based on the specified plot type.
        Parameters:
        plot_type (str): The type of plot to generate. It can be either 'training' or 'prediction'.
                         Default is 'training'.
                         - 'training': Plots the training parameters using the dataframe `self.df`.
                         - 'prediction': Plots the prediction parameters using data from a CSV file 
                                         located at `self.DataLocation` and specified by `self.aus_file`.
        Returns:
        None: This function displays the plot and does not return any value.
        """
        
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

    
        if plot_rainfall_only:
            fig, ax = plt.subplots(figsize=(8, 5))
            sc = ax.scatter(data['lon'], data['lat'], s=0.1, c=data['Rain mm/y'], cmap='Blues', alpha=0.5, vmax=1000)
            fig.colorbar(sc, ax=ax, label='Rainfall (mm/y)')
            ax.set_xlabel('Longitude ($\degree$E)')
            ax.set_ylabel('Latitude ($\degree$N)')
            ax.set_aspect('equal', 'box')
            plt.title('Rainfall Distribution')
            plt.show()


        if plot_recharge_only:
            fig, ax = plt.subplots(figsize=(8, 5))
            vmax = np.percentile(data['Recharge RC 50% mm/y'], 99)
            sc = ax.scatter(data['lon'], data['lat'], s=0.1, c=data['Recharge RC 50% mm/y'], cmap='viridis', alpha=0.5, vmax=vmax)
            fig.colorbar(sc, ax=ax, label='Recharge RC 50% (mm/y)')
            ax.set_xlabel('Longitude ($\degree$E)')
            ax.set_ylabel('Latitude ($\degree$N)')
            ax.set_aspect('equal', 'box')
            plt.title('Recharge RC 50% Distribution')
            plt.show()

    def compare_model_predictions(self, models=['rf', 'xgb']):
        """
        Compare the predictions of the specified models on the validation dataset.
        Parameters:
        models (list): List of models to compare. Options are 'rf' for Random Forest and 'xgb' for XGBoost. Default is ['rf', 'xgb'].
        This method checks if the specified models are trained and have made predictions. If not, it trains the models and generates predictions.
        It then creates a scatter plot comparing the predictions of the specified models against the actual values.
        The plot includes:
        - A scatter plot of the actual vs. predicted values for the specified models.
        - A 1:1 line indicating perfect predictions.
        - Labels for the x and y axes (Actual and Predicted).
        - A legend indicating which model's predictions are being plotted.
        Raises:
        AttributeError: If the models' predictions are not available and cannot be computed.
        """

        if 'rf' in models and not hasattr(self, 'rf_y_pred_valid'):
            self.RF_predictions()
        if 'xgb' in models and not hasattr(self, 'xgb_y_pred_valid'):
            self.XGB_predictions()

        fig, ax = plt.subplots(figsize=(8, 8))
        if 'rf' in models and 'xgb' in models:
            diff = (self.rf_y_pred_aus[self.y_predict] - self.xgb_y_pred_aus[self.y_predict]).abs() / self.xgb_y_pred_aus[self.y_predict] * 100
            print(max(diff))
            print(np.percentile(diff, 99))
            diff = (self.rf_y_pred_aus[self.y_predict] - self.xgb_y_pred_aus[self.y_predict]).abs() / self.xgb_y_pred_aus[self.y_predict] * 100
            max_diff = np.percentile(diff, 99)
            sc = ax.scatter(self.rf_y_pred_aus['lon'], self.rf_y_pred_aus['lat'], c=diff, cmap='coolwarm', alpha=0.5, vmax=max_diff)
            fig.colorbar(sc, ax=ax, label='Absolute Percentage Difference (RF - XGB)')

        ax.set_xlabel('Longitude ($\degree$E)')
        ax.set_ylabel('Latitude ($\degree$N)')
        ax.set_aspect('equal', 'box')
        plt.title('Difference between Random Forest and XGBoost Predictions')
        plt.show()
    def scatterplot(self, model='rf'):    
        """
        Creates a scatter plot of observed vs. predicted values for the specified model.
        
        Parameters:
        model (str): The model to use for predictions. Options are 'rf' (Random Forest), 
                    'xgb' (XGBoost), 'lasso' (Lasso Regression), or 'old_rf' (older Random Forest model).
        """
        
        if model == 'rf':
            if not hasattr(self, 'rf'):
                self.RF_train()
            holdout = pd.DataFrame({'obs': self.ytest, 'preds': self.rf.predict(self.Xtest)})
        
        elif model == 'xgb':
            if not hasattr(self, 'xgb'):
                self.XGB_train()
            holdout = pd.DataFrame({'obs': self.ytest, 'preds': self.xgb.predict(self.Xtest)})
        elif model == 'lasso':
            if not hasattr(self, 'lasso'):
                self.Lasso_train()
            holdout = pd.DataFrame({'obs': self.ytest, 'preds': self.lasso.predict(self.Xtest)})
        
        elif model == 'old_rf':
            if not hasattr(self, 'rf_old'):
                self.RF_train(old_model=True)
            holdout = pd.DataFrame({'obs': self.ytest, 'preds': self.rf_old.predict(self.Xtest)})
        else:
            raise ValueError("Model should be 'rf', 'xgb', 'lasso', 'old_rf'")



        plt.scatter(holdout['obs'], holdout['preds'], s=0.1)
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'{model.upper()} - Recharge rate')

        # Plot the line with the same limits
        plt.plot([1e-3, 1e4], [1e-3, 1e4], color='red', linestyle='--')

        # Set the same limits for both axes
        plt.xlim(1e-3, 1e4)
        plt.ylim(1e-3, 1e4)

        plt.xlabel('Observed Values [mm/y]')
        plt.ylabel('Predicted Values [mm/y]')
        plt.text(0.95, 0.05, f'R² = {round(r2_score(holdout["obs"], holdout["preds"]), 3)}\nRMSE = {round(np.sqrt(np.mean((holdout["preds"] - holdout["obs"])**2)), 3)}', 
                fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes,
                bbox=dict(facecolor='yellow', alpha=0.5))
        
        plt.show()

    def logplot(self):  

        loc_features = ['lat', 'lon', 'Rain mm/y', 'rainfall_seasonality', 
                'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 
                'ndvi_avg', 'clay_perc', 'soil_class']
        target_var = "Recharge RC 50% mm/y"

        X = self.df[loc_features]
        y = self.df[target_var]
        plt.figure(figsize=(10, 5))

        # Determine the common color range
        vmin = max(y.min(), 0.01)
        vmax = y.max()

        # Scatter plot
        sc = plt.scatter(X.lon, X.lat, c=y, s=0.5, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))

        # Set titles and labels
        plt.title('Recharge RC 50% distribution')
        plt.xlabel('Lon [°E]')
        plt.ylabel('Lat [°N]')

        # Add a colorbar
        cbar = plt.colorbar(sc, orientation='vertical', fraction=0.02, pad=0.1)
        cbar.set_label('Recharge Rate 50% mm/y')

        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Show plot
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Instantiate the Workflow class
    workflow = Workflow()
    # Plot the training parameters, focusing specifically on the recharge rate, using the training dataset.
    # workflow.plot_parameters(plot_type='training', plot_recharge_only=True)

    # Validate the models (Random Forest, XGBoost, and Lasso) on the validation dataset.
    # This method checks the overall performance of the models and prints the R² scores.
    # workflow.validate_models()  

    # Test the models (Random Forest, XGBoost, and Lasso) on the test dataset.
    # This evaluates how well the models perform on unseen data and prints the R² scores for the test dataset.
    # workflow.test_models()

    # Generate a scatter plot comparing observed values against predicted values, change input for desired model
    # workflow.scatterplot(model='xgb')

    # Train the Random Forest model with the updated parameters, enabling out-of-bag scoring and bootstrap sampling.
    # workflow.RF_train(oob_score=True,bootstrap=True)

    workflow.logplot()



