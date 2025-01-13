Text file with a short description of how each version of CNN works

Tip:
using the raster code you can create the images you want to pass to the model as auxiliary inputs. The raster code creates three different images for each variable, so you can only choose one of them. I would suggest to choose the images you want and copy them to the aux_inputs folder, from where they can be read directly from the CNN code

CNN_multichannels_version-1: In this version all the location inputs are stacked together and then concatenated with the output of each auxiliary channel. While counterintuitive, this version provides the best score of the CNN (up until now), when using only one image (Rain) and all the variables as location inputs - R2 score of a bit more than 0.70. The score drops if more auxiliary channels are combined
	suggestion: try different aux inputs in the 1 channel version to see if any of the variables fits better than rain
		    or try different combination of aux inputs in the multichannel version

CNN_multichannels_version-2: The multichannel version of the paper CNN. Each channel has its own auxiliary and location branch, which are concatenated together and then to the final version. It does not perform well for multiple channels.
	suggestion: again, try different combinations of inputs and see if some of them fit better to the code

CNN_multichannels_version-3: The auxiliary inputs are prossesed together as well as the location inputs and then they are concatenated. This version is to be run in colab, but with minor changes it can be run loacally or in runpod

Also, in the versions 2 and 3 there is routine with if's in case we want to pass only part of the variables as auxiliary data and all of them as location data. I was not able to make it work till now.