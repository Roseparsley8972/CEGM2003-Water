import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from datetime import datetime
from decimal import Decimal
import random
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataset import random_split
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#First attempt att getting CNN UNet model to work with data.

#Maybe move this function to its own place at some point
def normalize_dataset(dataset, scaler_x, scaler_y):
    min_x, max_x = scaler_x.data_min_[0], scaler_x.data_max_[0]
    min_y, max_y = scaler_y.data_min_[0], scaler_y.data_max_[0]
    normalized_dataset = []
     
    for idx in range(len(dataset)):
        x = dataset[idx][0]
        y = dataset[idx][1]
        norm_x = (x - min_x) / (max_x - min_x)
        norm_y = (y - min_y) / (max_y - min_y)
        
        #This is a stupid solution I believe, should be resolved in some other way
        normalized_dataset.append((torch.tensor(norm_x, dtype=torch.float32), torch.tensor(norm_y, dtype=torch.float32)))
        #normalized_dataset.append((norm_x, norm_y))



 
    return normalized_dataset



#Things that should be an input from a function
aus_file='Australia_grid_0p05_data.csv'
seed=42
y_var='Recharge RC 50% mm/y'


#Initalize data and parameters
start_time = datetime.now()
DataLocation = os.path.join(os.path.dirname(__file__), '..', 'data')
os.chdir(DataLocation)


train_params = ['Rain mm/y', 'rainfall_seasonality', 'PET mm/y', 'elevation_mahd', 'distance_to_coast_km', 'ndvi_avg', 'clay_perc', 'soil_class']

random.seed(seed)
random_num = random.randint(0, 1000)




#To the data I want to load, normalize and split into batches
train_dataset = pd.read_csv('train_data.csv')
test_dataset = pd.read_csv('test_data.csv')

filt_train = train_dataset[train_params + [y_var]].values
x_filt_train = train_dataset[train_params].values
y_filt_train = train_dataset[y_var].values.reshape(-1, 1)

filt_test = test_dataset[train_params + [y_var]].values
x_filt_test = test_dataset[train_params].values
y_filt_test = test_dataset[y_var].values.reshape(-1, 1)




#Normalize the data
scaler_x_train = MinMaxScaler()
scaler_y_train = MinMaxScaler()

scaler_x_train.fit(x_filt_train)
scaler_y_train.fit(y_filt_train)

# I need to just make all the data on a scale from 0 to 1.
#Using the same scaler is apparently standard
normalized_train_dataset = normalize_dataset(filt_train, scaler_x_train, scaler_y_train)

normalized_test_dataset = normalize_dataset(filt_test, scaler_x_train, scaler_y_train)

#print(f"Normalized Training Dataset: {normalized_train_dataset[:5]}")  # Print some of the normalized data


#So I want to use the train data for both validation and training. I'll use 80% of it for training and 20% for validation. 
#Then the test data is for testing the model on independent data. 

train_percnt = 0.8
train_size = int(train_percnt * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(normalized_train_dataset, [train_size, val_size])

#Now for the CNN model

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # First convolution
            nn.BatchNorm2d(out_channels),                                   # Batch normalization
            nn.ReLU(inplace=True),                                         # ReLU activation
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Second convolution
            nn.BatchNorm2d(out_channels),                                   # Batch normalization
            nn.ReLU(inplace=True)                                          # ReLU activation
        )

    def forward(self, x):
        return self.conv(x)  # Forward pass through the double convolution

# 8 in channels for the 8 parameters and 1 output. 
class UNet(nn.Module):
    def __init__(self,in_channels=8,out_channels=1,features=[64,128,256,512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList() #So contains how the model when activated should processed up, so how it should transform the data on the way up from flattened
        self.downs = nn.ModuleList() #And this is the same but for the path down. So it is a way to not have to write it all down I guess.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #And this is the Kernel that performs the pooling on the data, it downsises the data. Should make it half the size

        #Downsample path, so fill up the downs list
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        #Upsample path, so fill up the ups list
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))


        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downward path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) #Saving the data so that it can be used on the upward path
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Upward path
        skip_connections = skip_connections[::-1] #So now we are using the data saved earlier.
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


# Parameters for U-Net
in_channels = 3   # Assuming input channels are 3 (like RGB images)
out_channels = 1  # Typically 1 for binary segmentation
features = [64, 128, 256, 512]  # Adjustable features

# Model initialization
model = UNet(in_channels=in_channels, out_channels=out_channels, features=features).to(device)



#Training

def train_epoch(model, loader, optimizer, device):
    model.to(device)
    model.train() # specifies that the model is in training mode

    losses = []

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)

        # Model prediction
        preds = model(x)

        # MSE loss function
        loss = nn.MSELoss()(preds, y)

        losses.append(loss.cpu().detach())

        # Backpropagate and update weights
        loss.backward()   # compute the gradients using backpropagation
        optimizer.step()  # update the weights with the optimizer
        optimizer.zero_grad(set_to_none=True)   # reset the computed gradients

    losses = np.array(losses).mean()

    return losses

#Evaluation 
def evaluation(model, loader, device):
    model.to(device)
    model.eval() # specifies that the model is in evaluation mode

    losses = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            # Model prediction
            preds = model(x)

            # MSE loss function
            loss = nn.MSELoss()(preds, y)
            losses.append(loss.cpu().detach())

    losses = np.array(losses).mean()

    return losses


# Set training parameters
learning_rate = 0.001
batch_size = 64
num_epochs = 100

# Create the optimizer to train the neural network via back-propagation
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# Create the training and validation dataloaders to "feed" data to the model in batches
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(normalized_test_dataset, batch_size=batch_size, shuffle=False)


train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Train for one epoch
    train_loss = train_epoch(model, train_loader, optimizer, device)
    train_losses.append(train_loss)

    # Validate the model
    val_loss = evaluation(model, val_loader, device)
    val_losses.append(val_loss)

    # Print epoch results
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# Optionally, save the model at the end
torch.save(model.state_dict(), 'unet_model.pth')

# If needed, you can also add a testing phase to calculate results on the test set
test_loss = evaluation(model, test_loader, device)
print(f'Test Loss: {test_loss:.4f}')