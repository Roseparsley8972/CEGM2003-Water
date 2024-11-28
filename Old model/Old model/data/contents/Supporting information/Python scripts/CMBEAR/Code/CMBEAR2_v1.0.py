# -*- coding: utf-8 -*-
"""
Created on 22 September 2023 at 16:00

CMBEAR2_v1.0.py CMBEAR2 is a modified version of CMBEAR (Irvine and Cartwright, 2022)
that allows the user to choose the regular CMB method of estimating recharge,
and/or a stochastic approach which includes a runoff coefficient term in the
CMB calculation. When using the stochastic approach, for each point/bore that a
recharge estimate is needed, CMBEAR2 generates probability distributions for each
term in the CMB equation and calculates recharge 1,000 times. CMBEAR2 then
outputs the 5th, 50th and 95th percentiles of recharge (R5, R50 and R95). Climate
grids can be imported so that the average annual rainfall, PET or aridity index
at the location of bores can be extracted and appended to the recharge outputs
to facilitate further assessments.

@author: slee and dirvine
"""

# import numpy library and os library
import os
import numpy as np

# ===== Step 1: Set up and run the User command section =======================
#%%
#=========================  User commands  ===========================
datafolder = os.path.join(os.path.dirname(__file__), '..', 'InputData') 
outfolder  = os.path.join(os.path.dirname(__file__), '..', 'OutputFiles')
datafile   = 'Chloride_dataset_20230907_state_id.csv' # name of the excel workboook with the data
sheetname  = 'insert_sheet_name' # name of the excel worksheet containing the data if applicable              
print('User inputs read')

# ===== Step 2: Set up and run this section to set up the map =================
#%%


#=============== files===================
Dmean      = 'cl_deposition_mean.txt' # file name of file containing mean chloride depostion map
Dlower     = 'cl_deposition_5th.txt' # file name of file containing lower 95% of chloride deposition map (if applicable)
Dupper     = 'cl_deposition_5th.txt' # file name of file containing upper 95% of chloride deposition map (if applicable)
Dsd        = 'cl_deposition_sd.txt' # file name of file containing std dev of chloride deposition map (if applicable)
Dskew      = 'cl_deposition_skew.txt' # file name of file containing skew of chloride deposition map (if applicable)
Rc_ave     = 'RC.txt' # file name of file containing runoff coefficient annual average map (if applicable)
rainfile   = 'rain.txt' # file name of file containing long-term average rainfall map (if applicable)
aridfile   = 'aridity.txt' # file name of file containing long-term average rainfall map (if applicable)
petfile    = 'PET.txt' # file name of file containing long-term average PET map (if applicable)
#=============settings===================
useuncert  = 'yes' # a switch to include uncertainty analyses or not (also whether to read uncertainty in chloride deposition or not. Use either 'yes' or 'no')
useclim    = 'yes' # a switch to include/exclude the steps of the code relating to rainfall, aridity, and pet data. i.e. set to 'no' if no spatially variable climate data are available
userunoff  = 'yes' # a switch to include/exclude use of runoff coefficient in calculation of recharge.
useboth    = 'yes' # a switch to run both regular CMBEAR and CMBEAR2 (utilises runoff)
#========= map resolution/extent=========
res = 0.05       # resolution in degrees
latNmost = -44 # Northern most latitude of gridded map (centre of pixel)
lonEmost = 112 # Eastern most longitude of gridded map (centre of pixel)
nlat   = 681     # number of cells in N-S direction
nlon   = 841     # number of cells in E-W direction
headrows = 6     # number of header rows to skip
nodataval = -999 # No data values to be replaced with nan

#=======================================================================
# Load maps
#=======================================================================
os.chdir(datafolder) # change directory
if 'y' in useclim:
    print('Loading rain data')
    rain = np.loadtxt(rainfile,
                      skiprows=headrows)
    rain[rain == nodataval] = np.nan # replace nodatavals with NaN
    print('Loading aridity data')
    arid = np.loadtxt(aridfile,
                      skiprows=headrows)
    arid[arid == nodataval] = np.nan # replace nodatavals with NaN
    print('Loading pet data')
    pet = np.loadtxt(petfile,
                      skiprows=headrows)
    pet[pet == nodataval] = np.nan # replace nodatavals with NaN
else:
    print('No climate data provided')
print('Loading chloride deposition data')
ClD = np.loadtxt(Dmean, skiprows=headrows)
ClD[ClD == nodataval] = np.nan # replace nodatavals with NaN
if 'y' in useuncert:
    # upper95% chloride deposition
    ClDupper = np.loadtxt(Dupper,skiprows=headrows)
    ClDupper[ClDupper == nodataval] = np.nan # replace nodatavals with NaN
    # lower95% chloride deposition
    ClDlower = np.loadtxt(Dlower,skiprows=headrows)
    ClDlower[ClDlower == nodataval] = np.nan # replace nodatavals with NaN
    # Std chloride deposition
    ClDsd = np.loadtxt(Dsd,skiprows=headrows)
    ClDsd[ClDsd == nodataval] = np.nan # replace nodatavals with NaN
    # Skew chloride deposition
    ClDskew = np.loadtxt(Dskew,skiprows=headrows)
    ClDskew[ClDskew == nodataval] = np.nan # replace nodatavals with NaN
else:
    print('No upper/lower 95% bounds/std dev available for chloride deposition')
if 'y' in userunoff:
    print('Loading runoff coefficient data')
    rc = np.loadtxt(Rc_ave,
                      skiprows=headrows)
else:
    print('No runoff coefficient data provided')

# create arrays to search through maps to find appropriate pixel.
print('Building arrays for searching maps')
lats_map = np.ones(nlat)*latNmost
for i in range(1,nlat):
    lats_map[i] = lats_map[i-1]+res   
lats_map = lats_map[::-1]  # reverse order as search from top
lons_map = np.ones(nlon)*lonEmost
for i in range(1,nlon):
    lons_map [i] = lons_map[i-1]+res

print('Maps loaded')

# ===== Step 3: Next run this section to calculate recharge ===================
#%%
# import libraries
import numpy as np
import pandas as pd
import time
import os

os.chdir(datafolder) # change directory
print('CMBEAR running...\n=====================================')

# output excel file, accounting for .xls, .xlsx, .csv formats
if datafile.endswith('.xlsx') == True:
    temp= datafile.split('.xlsx')
    datfileout = temp[0]+'_out.xlsx'
if datafile.endswith('.xls') == True:
    temp= datafile.split('.xls')
    datfileout = temp[0]+'_out.xls'
if datafile.endswith('.csv') == True:
    temp= datafile.split('.csv')
    datfileout = temp[0]+'_out.csv'    

# Load data from the user supplied spreadsheet
os.chdir(datafolder) # change directory
print('Loading input spreadsheet')
if datafile.endswith('.csv') == True:
    Clgw_data = pd.read_csv(datafile)    
else:    
    Clgw_data = pd.read_excel(datafile,sheet_name=sheetname)

# QA/QC of data
cols = ['bore_id','reg_id','state_id','lat','lon','project','source','screen_top_m','screen_base_m','screen_mid_m','construction_type','bore_depth_m','hole_depth_m','sample_depth_m','my_depth_m','sample_date','Chloride'] #assign columns
Clgw = Clgw_data[cols] #execute the columns
Clgw = Clgw[~Clgw['Chloride'].isnull()]
print('Total chloride n=' + str(len(Clgw)))
Clgw = Clgw[~Clgw['lat'].isnull()] # ensure everything has a lat
Clgw = Clgw[~Clgw['lon'].isnull()] # ensure everything has a lon
Clgw = Clgw[~Clgw['my_depth_m'].isnull()] # ensure everything has a my_depth_m, otherwise we could get nested bores with unknown sample depths
#Clgw = Clgw[~Clgw['sample_date'].isnull()] # unhash this line if you want to ensure everything has a sample date, otherwise we can have some bores with double up measurements
Clgw = Clgw[~Clgw['Chloride'].str.contains('<', na=False)] #filter out any values that contain '<'
Clgw = Clgw[~Clgw['bore_id'].str.contains('<', na=False)] #filter out any values that contain '<'
Clgw['Chloride'] = Clgw['Chloride'].astype(str, errors = 'ignore').astype(float) #convert strings to floats
has_Cl = Clgw['Chloride'] > 0 # filter for bores w/Cl
Clgw = Clgw[has_Cl]
Clgw = Clgw.drop_duplicates(subset=['bore_id','reg_id','state_id','lat','lon','my_depth_m','sample_date','Chloride'], keep='first', inplace=False, ignore_index=False)
print('Total chloride after QA/QC n=' + str(len(Clgw)))
Clgw['reg_id'] = Clgw['reg_id'].fillna(0)
Clgw['reg_id'] = Clgw['reg_id'].astype(str) #convert to strings
os.chdir(outfolder) # change directory
Clgw.to_csv('Clgw_post_qa.csv', index=False) # export to analyse multiple Cl measurements
cols = ['bore_id','reg_id','state_id','lat','lon','my_depth_m','Chloride'] #assign columns
Clgw = Clgw[cols] #execute the columns

# use groupby function here to group up bores with multiple data entries
data_grp = Clgw.groupby(['bore_id','reg_id','state_id','lat','lon','my_depth_m']).describe() # Can add "percentiles = [0.05, 0.2, etc.]" inside brackets if needed.
os.chdir(outfolder) # change directory
data_grp.to_csv('Clgw_data_stats.csv', index=True) # export to analyse multiple Cl measurements

# Create the final chloride dataset after grouping
os.chdir(outfolder)
print('loading Clgw_data_stats.csv')
data_grp = pd.read_csv('Clgw_data_stats.csv')
os.chdir(outfolder)
print('loading Clgw_post_qa.csv')
Clgw = pd.read_csv('Clgw_post_qa.csv')
data_grp.iloc[1,6:] = data_grp.iloc[0,6:]
data_grp_2 = data_grp.rename(columns=data_grp.iloc[1]).loc[2:]
data_grp_2.reset_index(inplace=True, drop=True)
data = data_grp_2
columns_to_convert = ['lat','lon','my_depth_m','count','mean','std']
data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric)
cols = ['bore_id', 'lat', 'lon', 'mean', 'std', 'reg_id', 'state_id', 'my_depth_m', 'count']
Clgw = data[cols] # rearrange cols to right order
Clgw = Clgw.rename(columns={'mean':'Chloride_mean', 'std':'Chloride_sd'}) #rename col
os.chdir(outfolder) # change directory
Clgw.to_csv('Clgw_final.csv', index=False) # export to analyse multiple Cl measurements
print('Total chloride n=' + str(len(Clgw)))

# Grab bores with multiple measurements (>10) to calculate a blanket Coefficient of variation (CV)
# Tweak this part of code and adopt your custom CV if applicable
is_single = data['count'] == 1 #define filter as >10 as per Crosbie et al 2018 methods.
single_meas = data[is_single] #apply filter
print(len(single_meas))
multi_num = 10
is_multi = data['count'] > multi_num #define filter as >10 as per Crosbie et al 2018 methods.
multi_meas = data[is_multi] #apply filter
print(len(multi_meas))
os.chdir(outfolder) # change directory
multi_meas.to_csv('Multi_meas.csv', index=False)
multi_meas['cv'] = multi_meas['std'] / multi_meas['mean']
os.chdir(outfolder) # change directory
multi_meas.to_csv('Multi_meas_cv.csv', index=False)
m_cv = np.mean(multi_meas['cv'])
print('cv= ' + str(' %.2f'%m_cv))

# Prepare arrays for data storage
print('Preparing arrays to store data')
# create empty arrays to store the mapped C_P and Precip for the measured Clgw
if 'y' in userunoff:
    if 'y' in useclim:
        P_points     =   np.zeros(len(Clgw)) # array for rainfall values (from the map) that corresponds to each groundwater Cl value
        A_points     =   np.zeros(len(Clgw)) # array for aridity values (from the map) that corresponds to each groundwater Cl value
        PET_points   =   np.zeros(len(Clgw)) # array for PET values (from the map) that corresponds to each groundwater Cl value
    Cl_Pmean_points  =   np.zeros(len(Clgw)) # array for Cl in rainfall values (from the map) that corresponds to each groundwater Cl value
    Cl_Pstd_points   =   np.zeros(len(Clgw)) # array for Cl in rainfall values (from the map) that corresponds to each groundwater Cl value
    Cl_Pskew_points  =   np.zeros(len(Clgw)) # array for Cl in rainfall values (from the map) that corresponds to each groundwater Cl value
    rc_mean_points   =   np.zeros(len(Clgw)) # array for RC mean values
    lats_val         =   np.zeros(len(Clgw)) # will be used to plot the data points on the map later
    lons_val         =   np.zeros(len(Clgw)) # will be used to plot the data points on the map later
    R_dist           =   np.zeros(1000) # will be used to write all 1000 R values temporarily while iterating through each row of gw Cl dataset
    R_rc_50        =   np.zeros(len(Clgw)) # will be used to write R median data calculated from rc CMB eq.
    R_rc_mean        =   np.zeros(len(Clgw)) # will be used to write R mean data calculated from rc CMB eq.
    R_rc_5       =   np.zeros(len(Clgw)) # will be used to write R lower data calculated from rc CMB eq.
    R_rc_95       =   np.zeros(len(Clgw)) # will be used to write R upper data calculated from rc CMB eq.
if 'y' in useboth:
    Cl_Pupper_points =   np.zeros(len(Clgw)) # array for Cl in rainfall values (from the map) that corresponds to each groundwater Cl value
    Cl_Plower_points =   np.zeros(len(Clgw)) # array for Cl in rainfall values (from the map) that corresponds to each groundwater Cl value
    R_mean           =   np.zeros(len(Clgw)) # will be used to write R mean data
    R_5          =   np.zeros(len(Clgw)) # will be used to write R lower data
    R_95          =   np.zeros(len(Clgw)) # will be used to write R upper data
else:
    if 'y' in useclim:
        P_points     =   np.zeros(len(Clgw)) # array for rainfall values (from the map) that corresponds to each groundwater Cl value
        A_points     =   np.zeros(len(Clgw)) # array for aridity values (from the map) that corresponds to each groundwater Cl value
        PET_points   =   np.zeros(len(Clgw)) # array for PET values (from the map) that corresponds to each groundwater Cl value
    Cl_Pmean_points  =   np.zeros(len(Clgw)) # array for Cl in rainfall values (from the map) that corresponds to each groundwater Cl value
    Cl_Pupper_points =   np.zeros(len(Clgw)) # array for Cl in rainfall values (from the map) that corresponds to each groundwater Cl value
    Cl_Plower_points =   np.zeros(len(Clgw)) # array for Cl in rainfall values (from the map) that corresponds to each groundwater Cl value
    lats_val         =   np.zeros(len(Clgw)) # will be used to plot the data points on the map later
    lons_val         =   np.zeros(len(Clgw)) # will be used to plot the data points on the map later

# Generate distributions for ClDep and RC
print('Generate distributions')
start = time.time() # start a timer
#Functions to generate PearsonIII, normal and uniform distributions
def generate_pearsonIII(Dmean, Dstd_dev, Dskew, n, seed):
    np.random.seed(seed)
    # Ensure parameters are valid
    Dstd_dev = max(Dstd_dev, 0.001)
    Dskew = Dskew if Dskew != 0 else 0.001
    # Check for NaN values
    if np.isnan(Dmean) or np.isnan(Dstd_dev) or np.isnan(Dskew):
        raise ValueError("One or more input values are NaN. No D distribution possible")
    # Generate n standard normal deviates
    z = np.random.normal(size=num_values)  
    # Compute frequency factor for each z
    KY = (2 / Dskew) * ((((z - (Dskew / 6)) * (Dskew / 6) + 1)**3) - 1)
    # Compute replicates according to Pearson Type III distribution
    replicates = np.abs(Dmean + KY * Dstd_dev)
    return replicates

def generate_normal(mean, std_dev, num_values, seed):
    np.random.seed(seed)
    values = np.abs(np.random.normal(mean, std_dev, num_values))
    return values

def generate_uniform(low, high, num_values, seed3):
    np.random.seed(seed3)
    values = np.random.uniform(low, high, num_values)
    return values
# Parameters constants
num_values = 1000
seed = 1
seed2 = 3100
seed3 = 2
low = 0.33
high = 0.66

#the loop to generate distributions and calculate R values
if 'y' in userunoff:
    print('Running CMB equation with runoff coefficient to calculate recharge...')  
    # loop to prepare the rainfall and chloride deposition values
    for i in range(0,len(Clgw)):
        print('Starting ' + str(i+1) + ' out of ' + str(len(Clgw)) + ' measurements')
        lats_val[i] = np.argmax(lats_map-(res/2)<=Clgw.iloc[i,1]) 
        lons_val[i] = np.argmax(lons_map+(res/2)>=Clgw.iloc[i,2])
        lats_val_temp = np.argmax(lats_map-(res/2)<=Clgw.iloc[i,1]) 
        lons_val_temp = np.argmax(lons_map+(res/2)>=Clgw.iloc[i,2])  
        # next extract the Cl in precip mean, std dev, mean precip, and RC values
        if 'y' in useclim:
            P_points[i]       = rain[lats_val_temp, lons_val_temp] # grab the P values
            A_points[i]       = arid[lats_val_temp, lons_val_temp] # grab the Aridity values
            PET_points[i]     = pet[lats_val_temp, lons_val_temp] # grab the PET values
        Cl_Pmean_points[i]    = ClD[lats_val_temp, lons_val_temp] # grab the Cl dep mean values
        if 'y' in useboth:
            Cl_Pupper_points[i]   = ClDupper[lats_val_temp, lons_val_temp] # grab the Cl dep upper values
            Cl_Plower_points[i]   = ClDlower[lats_val_temp, lons_val_temp] # grab the Cl dep lower values
        Cl_Pstd_points[i]     = ClDsd[lats_val_temp, lons_val_temp] # grab the Cl dep std dev values
        Cl_Pskew_points[i]    = ClDskew[lats_val_temp, lons_val_temp] # grab the Cl dep skew values
        rc_mean_points[i]     = rc[lats_val_temp, lons_val_temp] # grab the rc mean values
        #Cl P distribution
        Dmean = Cl_Pmean_points[i]
        Dstd_dev = Cl_Pstd_points[i]
        Dskew = Cl_Pskew_points[i]
        Cl_P_dist = generate_pearsonIII(Dmean, Dstd_dev, Dskew, num_values, seed)
        #Cl gw distribution
        mean = Clgw.iloc[i,3]
        std_dev = Clgw.iloc[i,3] * m_cv
        Cl_gw_dist = generate_normal(mean, std_dev, num_values, seed2)
        #alpha distribution
        alpha_dist = generate_uniform(low, high, num_values, seed3)
        #R distribution (calculations, iterating through all 1000 rows in each variable)
        for j in range(0,len(R_dist)):
            R_dist[j] = np.asarray(((100*Cl_P_dist[j]*(1-(alpha_dist[j]*rc_mean_points[i]))))/Cl_gw_dist[j])
        if 'y' in useboth:
            R_rc_50[i] = np.percentile(R_dist, 50)
            R_rc_mean[i] = np.mean(R_dist)
            R_rc_5[i] = np.percentile(R_dist, 5)
            R_rc_95[i] = np.percentile(R_dist, 95)
            
            R_mean = np.asarray(100*Cl_Pmean_points/Clgw.iloc[:,3])
            if 'y' in useuncert:
                R_95 = np.asarray(100*Cl_Pupper_points/Clgw.iloc[:,3])
                R_5 = np.asarray(100*Cl_Plower_points/Clgw.iloc[:,3])
        else:
            R_rc_50[i] = np.percentile(R_dist, 50)
            R_rc_mean[i] = np.mean(R_dist)
            R_rc_5[i] = np.percentile(R_dist, 5)
            R_rc_95[i] = np.percentile(R_dist, 95)
else:
    print('Running regular CMB equation to calculate recharge...')
    # loop to prepare the rainfall and chloride deposition values
    for i in range(0,len(Clgw)):
        lats_val[i] = np.argmax(lats_map-(res/2)<=Clgw.iloc[i,1]) 
        lons_val[i] = np.argmax(lons_map+(res/2)>=Clgw.iloc[i,2])
        lats_val_temp = np.argmax(lats_map-(res/2)<=Clgw.iloc[i,1]) 
        lons_val_temp = np.argmax(lons_map+(res/2)>=Clgw.iloc[i,2])  
        # next extract the Cl in precip, and mean precip values
        if 'y' in useclim:
            P_points[i]       = rain[lats_val_temp, lons_val_temp] # grab the P values
            A_points[i]       = arid[lats_val_temp, lons_val_temp] # grab the Aridity values
            PET_points[i]     = pet[lats_val_temp, lons_val_temp] # grab the PET values
            rc_mean_points[i] = rc[lats_val_temp, lons_val_temp] # grab the RC values
        Cl_Pmean_points[i]    = ClD[lats_val_temp, lons_val_temp]
        if 'y' in useuncert:
            Cl_Pupper_points[i]   = ClDupper[lats_val_temp, lons_val_temp]
            Cl_Plower_points[i]   = ClDlower[lats_val_temp, lons_val_temp]    
    # Eqn 4 of Davies and Crosbie has a different recharge equation...
    R_mean = np.asarray(100*Cl_Pmean_points/Clgw.iloc[:,3])
    if 'y' in useuncert:
        R_95 = np.asarray(100*Cl_Pupper_points/Clgw.iloc[:,3])
        R_5 = np.asarray(100*Cl_Plower_points/Clgw.iloc[:,3])
end = time.time() #=== all finished, calculate run time
#================= end
print("Finished. Run time ",'%.2g'%(end - start)," seconds")
print('=====================================\n')
#%%
# ===== Step 4: write the resulting values at the end of the input sheet=======

if 'y' in userunoff:
    print('Writing outputs to spreadsheet')
    Rdata = Clgw.iloc[:, 0:9]
    if 'y' in useclim:
        Rdata['Rain mm/y'] = P_points
        Rdata['Aridity'] = A_points
        Rdata['PET mm/y'] = PET_points
        Rdata['RC'] = rc_mean_points
    if 'y' in useboth:
        Rdata['Deposition mean kg/ha/yr'] = Cl_Pmean_points
        if 'y' in useuncert:
            Rdata['Deposition 95% kg/ha/yr'] = Cl_Pupper_points
            Rdata['Deposition 5% kg/ha/yr'] = Cl_Plower_points
        Rdata['Recharge RC 50% mm/y']       = R_rc_50
        Rdata['Recharge RC mean mm/y']       = R_rc_mean
        Rdata['Recharge mean mm/y']       = R_mean
        if 'y' in useuncert:
            Rdata['Recharge RC 95% mm/y']       = R_rc_95
            Rdata['Recharge RC 5% mm/y']       = R_rc_5
            Rdata['Recharge 95% mm/y']       = R_95
            Rdata['Recharge 5% mm/y']       = R_5
        # Calculate R/P ratio
        Rdata['Rrc/P'] = Rdata['Recharge RC 50% mm/y'] / Rdata['Rain mm/y']  
        Rdata['R/P'] = Rdata['Recharge mean mm/y'] / Rdata['Rain mm/y']
        # reaarange columns order
        if 'y' in useclim:
            cols = ['bore_id', 'reg_id', 'state_id', 'lat', 'lon', 'my_depth_m', 'Aridity', 'PET mm/y', 'RC', 'Rain mm/y', 'Deposition mean kg/ha/yr', 'Deposition 95% kg/ha/yr', 'Deposition 5% kg/ha/yr', 'Chloride_mean', 'count', 'Recharge mean mm/y', 'Recharge 95% mm/y', 'Recharge 5% mm/y', 'R/P', 'Recharge RC 50% mm/y', 'Recharge RC mean mm/y', 'Recharge RC 95% mm/y', 'Recharge RC 5% mm/y', 'Rrc/P']
            Rdata = Rdata[cols]
    else:
        Rdata['Recharge mean mm/y']       = R_rc_mean
        if 'y' in useuncert:
            Rdata['Recharge 95% mm/y']       = R_rc_95
            Rdata['Recharge 5% mm/y']       = R_rc_5
        # Calculate R/P ratio
        Rdata['Rrc/P'] = Rdata['Recharge RC 50% mm/y'] / Rdata['Rain mm/y']
        # reaarange columns order
        if 'y' in useclim:
            cols = ['bore_id', 'reg_id', 'state_id', 'lat', 'lon', 'my_depth_m', 'Aridity', 'PET mm/y', 'RC', 'Rain mm/y', 'Chloride_mean', 'count', 'Recharge RC 50% mm/y', 'Recharge RC mean mm/y', 'Recharge RC 95% mm/y', 'Recharge RC 5% mm/y', 'Rrc/P']
            Rdata = Rdata[cols]
else:
    print('Writing outputs to spreadsheet')
    Rdata = Clgw.iloc[:, 0:9]
    if 'y' in useclim:
        Rdata['Rain mm/y'] = P_points
        Rdata['Aridity'] = A_points
        Rdata['PET mm/y'] = PET_points
        Rdata['RC'] = rc_mean_points
    Rdata['Deposition mean kg/ha/yr'] = Cl_Pmean_points
    if 'y' in useuncert:
        Rdata['Deposition 95% kg/ha/yr'] = Cl_Pupper_points
        Rdata['Deposition 5% kg/ha/yr'] = Cl_Plower_points
    Rdata['Recharge mean mm/y']       = R_mean
    if 'y' in useuncert:
        Rdata['Recharge 95% mm/y']       = R_95
        Rdata['Recharge 5% mm/y']       = R_5
    # Calculate R/P ratio
    Rdata['R/P'] = Rdata['Recharge mean mm/y'] / Rdata['Rain mm/y']
    # reaarange columns order
    if 'y' in useclim:
        cols = ['bore_id', 'reg_id', 'state_id', 'lat', 'lon', 'my_depth_m', 'Aridity', 'PET mm/y', 'RC', 'Rain mm/y', 'Deposition mean kg/ha/yr', 'Deposition 95% kg/ha/yr', 'Deposition 5% kg/ha/yr', 'Chloride_mean', 'count', 'Recharge mean mm/y', 'Recharge 95% mm/y', 'Recharge 5% mm/y', 'R/P']
        Rdata = Rdata[cols]
Rdata = Rdata.rename(columns={'count':'Chloride_count'}) #rename col
# Some points maybe outside the coverage of RC dataset (e.g. on Green Island), these show up as nans in the col Recharge RC mean mm/y 
print("before: " + str(len(Rdata))) #prints number of bores before removing nan values from Recharge RC
Rdata = Rdata[~Rdata['Recharge RC mean mm/y'].isnull()]
print("after: " + str(len(Rdata)))
if '.csv' in datfileout:
    Rdata.to_csv(datfileout, index=False) 
else:    
    Rdata.to_excel(datfileout, index=False) # add to existing input file
end = time.time() #=== all finished, calculate run time

#================= end
print("Outputs written to spreadsheet "+datfileout)
print("Finished. Run time ",'%.2g'%(end - start)," seconds")
print('=====================================\n')