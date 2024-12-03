import os
import joblib
import pandas as pd

# Paths
models_folder = os.path.join(os.path.dirname(__file__), '..', 'Trained_models')
data_folder = os.path.join(os.path.dirname(__file__), '..', 'Data')
test_data_file = os.path.join(data_folder, 'test_data.csv')

# Load test data
test_data = pd.read_csv(test_data_file)
X_test = test_data[['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']]
y_test = test_data['Recharge RC 50% mm/y']

# Iterate over all models in the models folder
for model_file in os.listdir(models_folder):
    if model_file.endswith('.pkl'):
        model_path = os.path.join(models_folder, model_file)
        
        # Load the model
        model = joblib.load(model_path)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate R2 score
        r2 = model.score(X_test, y_test)
        
        # Print the R2 score
        print(f'Model: {model_file}, R2 Score: {r2:.4f}')