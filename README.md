# CEGM2003-Water

The repository of team Water, containing the data, code and outputs of the different approaches we used for groundwater recharge estimation across Australia. 

Important folders:

* Old Model

    The original Random Forest for groundwater estimation, as described in Lee et al. (2024) 

- New Model

  The folder contains all the different approaches we tried for estimating groundwater in Australia:

  - **Improved_RF_code**
    - Contains the optimization of the original Random Forest
    - Includes optimization of hyperparameters and computational time

  - **XgBoost**
    - Our alternative to the Random Forest
    - Includes code for optimizing its hyperparameters

  - **CNN_from_R**
    - Contains a deep-neural network approach for estimating groundwater recharge in Australia based on Kirkwood et al. (2022)
    - The network contains two branches:
      - **Convolutional branch**: Processes raster images containing information for geological values
      - **Fully connected branch**: Receives the same features as the RF and XgBoost along with location coordinates
    - **Key Details**:
      - Images are created using `raster.ipynb` in the `rastering_code` folder
      - These images are stored in the `images_for_CNN` folder
      - To feed images into `CNN_final.ipynb`, they must be moved to the `aux_inputs` folder inside `CNN_from_R`
      - `CNN_final.ipynb` extracts the images and creates **32x32 images** centered at each measurement site
      - These centered images serve as the input for the convolutional branch
      - The network supports multiple image channels in the convolutional branch
      - Training time varies (30 minutes to 1 hour), depending on:
        - Number of channels
        - Epochs
        - Batch size
      - Requires a GPU for efficient training