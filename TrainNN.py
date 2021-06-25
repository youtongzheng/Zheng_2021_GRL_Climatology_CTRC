# setting up everything
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

f  = xr.open_dataset("/lustre/ytzheng/Data/CTRC_Yannian/All_2014_NCEP_1stprcesd.nc")
f = f.drop(var_names=['SW_heat_dmean']).drop(var_names=['LWP']).drop(var_names=['CF'])

print('Finish reading the preprocessed data and start preparing data for training...')

# STEP1: seperate input and output
varflag = []
out_lst = ['CTRC','SW_heat','LW_cool','LW_heat','LW_CRE']

for var in f.var_names.values:
    if var in out_lst:
        varflag.append('output')
    else:
        varflag.append('input')
        
f = f.assign_coords(varflag=("var_names", varflag))

fin = f.where(f.varflag == 'input', drop = True)
fout = f.where(f.varflag == 'output', drop = True)
output_tmp = fout.arrvar.values

var_in = fin.var_names.values
var_out = fout.var_names.values
nvar_in = fin.var_names.size
nvar_out = fout.var_names.size

#add solz as input 
input_tmp = np.concatenate((fin.arrvar.values, np.expand_dims(fin.solzc.values, axis=1)), axis=1)

nvar_in = nvar_in + 1
var_in = np.append(var_in, 'solz')

#STEP2: normalize (or standardize) the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
input_tmp = scaler.fit_transform(input_tmp)

from pickle import dump
dump(scaler, open('scaler.pkl', 'wb'))

#Step3: combine them and shuffle
tmp = np.concatenate([input_tmp, output_tmp], axis=1)
np.random.shuffle(tmp)

#Step4:seperate them into input and output, and into train and validation 
train_tmp, val_tmp = np.array_split(tmp, 2, axis = 0)

train_input = train_tmp[:,0:nvar_in]
train_output = train_tmp[:,nvar_in:nvar_in + nvar_out]

val_input = val_tmp[:,0:nvar_in]
val_output = val_tmp[:,nvar_in:nvar_in + nvar_out]

#Let's train!
print('Finish preparing the data and start training...')

from BuildNN import *
model.fit(train_input, train_output, epochs=5)

#save the model
model.save("my_h5_model.h5")

#save the data for visualization/validation
print('Saving the data')
ds_val_input = xr.DataArray(val_input, coords=[np.arange(val_input.shape[0]) , var_in],
                       dims=["sample", "var_names"])
ds_val_input = ds_val_input.to_dataset(name = 'arrvar')
ds_val_input.to_netcdf("/lustre/ytzheng/Data/CTRC_Yannian/All_2014_NCEP_val_input.nc")

ds_val_output = xr.DataArray(val_output, coords=[np.arange(val_input.shape[0]) , var_out],
                       dims=["sample", "var_names"])
ds_val_output = ds_val_output.to_dataset(name = 'arrvar')

ds_val_output.to_netcdf("/lustre/ytzheng/Data/CTRC_Yannian/All_2014_NCEP_val_output.nc")

print('Done!!')
