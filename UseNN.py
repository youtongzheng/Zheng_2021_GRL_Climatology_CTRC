import numpy as np
import xarray as xr
import datetime

f  = xr.open_dataset("/lustre/ytzheng/Data/CTRC_Yannian/All_2014_NCEP_1stprcesd.nc")

f = f.drop(var_names=['CTRC']).drop(
    var_names=['SW_heat']).drop(
    var_names=['LW_cool']).drop(
    var_names=['LW_heat']).drop(
    var_names=['LW_CRE']).drop(
    var_names=['LWP']).drop(
    var_names=['CF']).drop(
    var_names=['SW_heat_dmean'])

input_tmp = np.concatenate((f.arrvar.values, np.expand_dims(f.solzc.values, axis=1)), axis=1)

# scaling
from pickle import load
scaler = load(open('scaler.pkl', 'rb'))
input_tmp = scaler.transform(input_tmp)

# applying the NN to predict
import sys, subprocess
out = subprocess.getoutput('/bin/tcsh -c "module load tensorflow && printenv PYTHONPATH"')
sys.path = out.split("\n")[-1].split(":") + sys.path
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('my_h5_model.h5')

prediction = model.predict(input_tmp)

#prepare for output
ds_prediction = xr.DataArray(prediction, coords=[f.sample.values, ['CTRC', 'SW_heat', 'LW_cool',
       'LW_heat', 'LW_CRE']],dims=["sample", "var_names"])
ds_prediction = ds_prediction.assign_coords(lat=("sample", f.lat.values))
ds_prediction = ds_prediction.assign_coords(lon=("sample", f.lon.values))
ds_prediction = ds_prediction.assign_coords(lat0=("sample", f.lat0.values))
ds_prediction = ds_prediction.assign_coords(lon0=("sample", f.lon0.values))
ds_prediction = ds_prediction.assign_coords(julian_day=("sample", f.julian_day.values))
ds_prediction = ds_prediction.assign_coords(UTC=("sample", f.UTC.values))
ds_prediction = ds_prediction.assign_coords(scale=("sample", f.scale.values))
ds_prediction = ds_prediction.assign_coords(solzc=("sample", f.solzc.values))
ds_prediction = ds_prediction.to_dataset(name = 'arrvar')

ds = xr.concat([ds_prediction.arrvar, (ds_prediction.sel(var_names = 'SW_heat')*ds_prediction.scale).assign_coords(var_names = 'SW_heat_dmean').arrvar],"var_names")
ds = ds.to_dataset()

#convert julian day to date
date_list = [datetime.datetime.strptime(str(int(14000 + i)), '%y%j').date() for i in ds.julian_day.values]
ds = ds.assign_coords(time = ("sample", np.array(date_list, dtype = 'datetime64')))

#output
out_fn = "/lustre/ytzheng/Data/CTRC_Yannian/All_2014_NCEP_1stprcesd_prediction.nc"
ds.to_netcdf(out_fn)

print('Done!')