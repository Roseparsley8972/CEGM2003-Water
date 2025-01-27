# CEGM2003-Water

The repository of team Water, containing the data, code and outputs of the different approaches we used for groundwater recharge estimation across Australia. 

Important folders:

* Old Model

    The original Random Forest for groundwater estimation, as described in Lee et al. (2024) 

* New Model

The folder contains all the different approaches we tried for estimating groundwater in Australia:
    
        * Improved_RF_code contains the optimization of the original Random Forest, with respect to the hyperparameters and the computational time

        * XgBoost is our alternative to the Random Forest, together with the code for optimizing its hyperparameters

        * CNN_from_R is contains a deep-neural network approach for the estimation of groundwater recharge in Australia based on Kirkwood et al (2022). The network contains two branches, one that implements a set of convolutions for processing raster images containing information for geological values and a fully connected one which receives the same features as the RF and XgBoost together with their location coordinates. The images where created by us using the raster.ipynb in *rastering_code*. The code creates multiple images of each target variable the user defines and stores them in *images_for_CNN*. The ones are going to be fed in the CNN_final.ipynb have to be stored in the *aux_inputs* folder inside CNN_from_R. Then, CNN_final.ipynb extracts the images and creates 32x32 images centered in the the site of each measurement. These new centered images serve as the input of the convolutional branch. 
        The network is designed in a way that multiple image channels can be implemented in the convolutional branch. It takes 30 min to around an hour, given the number of channels, the epochs and the batch size, to be trained in a GPU.   