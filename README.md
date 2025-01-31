# CEGM2003-Water

The repository of group Water, containing the data, code and outputs of the different approaches we used for groundwater recharge estimation across Australia. 

Important folders:

* Old Model

    The original code and datase for groundwater estimation Random Forest, as described in Lee et al. (2024) 

* New Model

  The folder contains all the different approaches we tried for estimating groundwater in Australia:

  - **Workflow.py**
    - This code contains a summary of the Lasso, improved RF and XGBoost models. Different functions can be called to train and test the models on the same datasets, as well as to plot y-y plots and predictions of recharge rate in Australia. It combines the main aspects of all the work done for all three models, including those found in the "Improved_RF_code" folder and the "XGBoost" folder.
    - Further details can be found in the file "Workflow.md"

  - **Hyperparameter_optimization** 
    - Contains code to find optimal hyperparameters used in Workflow for the 3 different models  

  - **Improved_RF_code**
    - Contains the original codes used before the Workflow and Hyperparameter_optimization code was made for the optimization of the original Random forest
    
  - **XgBoost**
    - Contains the original codes used before the Workflow and Hyperparameter_optimization code was made for the testing and optimization of XgBoost.
    - in xgboost_hyperparameter_tuning.ipynb a more methodic approach to hyperparamter tuning was attempted, which did not lead to better results but is included for completeness. This code also includes feature importance, which was not considered improtant enough to include in Workflow.py

  - **CNN_from_R**
    - Contains a deep-neural network approach for estimating groundwater recharge in Australia based on Kirkwood et al. (2022)
    - **File/Folder Description**
      - CNN_final.ipynb: notebook used to train the NN. It has been run with Tensorflow v2.18.0 using the GPU on Google Colab.
      - CNN_paper.R: Original code used in Kirkwood et al. (2022)
      - rastering_code: contains code that creates raster images
      - images_for_CNN: contains raster images that can be used for input. The images with the name "..._bound.tif" have an upper bound in the value. The reason for this is that the raster format has a maximum range of values, and had very high values in a limited region, which decreased the available range of values in all the other regions.
      - aux_inputs: this folder contains the images that are taken as input to to CNN_final.ipynb code.
      - models: contains the weights of the trained models

  - **Data**: contains the training, validation and test datasets, the original unsplit dataset (dat07_u.csv) and the file with unseen data used for predictions(Australia_grid_0p05_data_with_rain.csv). It also contains the recharge rate predictions given by different models.

  - **Splitting data**
    - code used to split the data into train, validation and test

  - **Trained_models**
    - contains the trained models for Lasso, RF and XGBoost

  - **Rain**
    - contains the code used to fix the original dataset, which contained incorrent rainfall data.

* **Requirements_workflow.txt**
  - contains the packages and the versions that were used to run and produce the results of Workflow.py

* **Requirements_CNN.txt**
  - contains the packages and the versions that were used to run and produce the results of CNN_final.ipynb

  