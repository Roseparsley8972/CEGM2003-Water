# Instructions for Using Workflow.py

## Overview
`Workflow.py` is a script designed to facilitate a machine learning workflow for predicting recharge rates using Random Forest and XGBoost models. The script includes methods for loading data, training models, performing cross-validation, generating predictions, and visualizing results.

## Class: Workflow

### Initialization
To initialize the `Workflow` class, you can specify the following parameters:
- `k_num` (int): Number of folds for cross-validation. Default is 10.
- `y_var` (str): Target variable for training. Default is 'Recharge RC 50% mm/y'.
- `y_predict` (str): Name of the prediction column. Default is 'R50'.
- `aus_file` (str): Filename of the Australian grid data. Default is 'Australia_grid_0p05_data.csv'.
- `seed` (int): Random seed for reproducibility. Default is 42.
- `test_data` (bool): Flag to indicate if pre split data (True) or the split from the original study (False) should be used

### Validation
To validate the models, you can use the following methods:

#### Random Forest Cross-Validation
To perform cross-validation on the Random Forest model, use the `RF_cross_validdation` method:
```python
workflow.RF_cross_validdation()
```
This method will print the R2 score, RMSE, and MAE for each fold.

#### XGBoost Cross-Validation
To perform cross-validation on the XGBoost model, use the `XGB_cross_validation` method:
```python
workflow.XGB_cross_validation()
```
This method will print the R2 score, RMSE, and MAE for each fold.

#### Validate Models
To validate the performance of the trained models on the validation dataset, use the `validate_models` method:
```python
workflow.validate_models(model='all')
```
You can specify the model to validate by setting the `model` parameter to 'rf' for Random Forest, 'xgb' for XGBoost, or 'all' to validate both models. This method will print the R2 score for the specified model(s).


## Instructions for Using the Workflow Class for Plotting Predictions and Parameters
The `Workflow` class provides methods to train machine learning models, generate predictions, and plot the results. Below are the instructions for using the class to plot model predictions and parameters.

### Plotting Model Predictions

To plot the predictions of a trained model, use the `plot_model_predictions` method. This method creates a scatter plot of the predictions with longitude and latitude on the x and y axes, respectively. The color of the points represents the predicted recharge rate.

#### Parameters:
- `model` (str): The model to use for predictions. Options are 'rf' for Random Forest and 'xgb' for XGBoost. Default is 'rf'.

### How to use `plot_parameters`

The `plot_parameters` function generates scatter plots of parameters based on the specified plot type. It can plot either training parameters or prediction parameters.

#### Parameters:
- `plot_type` (str): The type of plot to generate. It can be either 'training' or 'prediction'. Default is 'training'.
    - `'training'`: Plots the training parameters using the dataframe `self.df`.
    - `'prediction'`: Plots the prediction parameters using data from a CSV file located at `self.DataLocation` and specified by `self.aus_file`.

#### Returns:
- None: This function displays the plot and does not return any value.

#### Example Usage:
###

