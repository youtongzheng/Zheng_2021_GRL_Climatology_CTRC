import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import glob
from tqdm import tqdm

in_fns = glob.glob("/lustre/ytzheng/Data/CTRC_Yannian/*2014_NCEP_month*withCF.nc")
ds  = xr.open_mfdataset(in_fns, decode_times=False, decode_cf=False, concat_dim='ncase', combine='nested')

print('Finish reading raw data, start process...')

#assigne variable names
MODISvarname = ['lat', 'lon', 'year','julian_day','UTC','SST','solz','CTT',
          'COD','LWP','re','t2m','td','rh','CTRC','SW_heat','LW_cool','LW_heat','LW_CRE','CF']

NCEPvarname = ['TMP_100mb','RH_100mb',
           'TMP_200mb','RH_200mb',
           'TMP_300mb','RH_300mb',
           'TMP_400mb','RH_400mb',
           'TMP_500mb','RH_500mb',
           'TMP_600mb','RH_600mb',
           'TMP_700mb','RH_700mb',
           'TMP_800mb','RH_800mb',
           'TMP_900mb','RH_900mb',
           'TMP_1000mb','RH_1000mb']

varname = MODISvarname + NCEPvarname

ds = ds.rename({'nvar':'var_names', 'ncase':'sample'})
ds['var_names'] = np.array(varname).astype('object')

#set some input variables as coordinate  (e.g. geolocation and time)
ind = np.where(ds.var_names.values == 'lat')[0]
tmp = ds.isel(var_names=ind)
tmp = tmp.arrvar.values[:,0]
ds = ds.assign_coords(lat=("sample", tmp))

ind = np.where(ds.var_names.values == 'lon')[0]
tmp = ds.isel(var_names=ind)
tmp = tmp.arrvar.values[:,0]
ds = ds.assign_coords(lon=("sample", tmp))

ind = np.where(ds.var_names.values == 'julian_day')[0]
tmp = ds.isel(var_names=ind)
tmp = tmp.arrvar.values[:,0]
ds = ds.assign_coords(julian_day=("sample", tmp))

ind = np.where(ds.var_names.values == 'UTC')[0]
tmp = ds.isel(var_names=ind)
tmp = tmp.arrvar.values[:,0]
ds = ds.assign_coords(UTC=("sample", tmp))

ds = ds.assign_coords(lat0=("sample", np.round(ds.lat.values)))
ds = ds.assign_coords(lon0=("sample", np.round(ds.lon.values)))

var_range = (5, 40)
ds = ds.isel(var_names=slice(*var_range, 1))

#convert julian day to date
print('Converting julian day to date...')
import datetime
date_list = [datetime.datetime.strptime(str(int(14000 + i)), '%y%j').date() for i in ds.julian_day.values]
ds = ds.assign_coords(time = ("sample", np.array(date_list, dtype = 'datetime64')))

#compute the solar zenith angle
print('Computing instant solar zenith angles...')
from utils import zenith

solzc = (zenith(ds.julian_day - 1. + ds.UTC/24., ds.lat, ds.lon))*180./(np.pi)
solzc = xr.where(solzc <= 90., solzc, 90.)
ds = ds.assign_coords(solzc=("sample", solzc))

# compute the daily mean solar zenith angle
print('Computing daily mean solar zenith angles...')
offset = np.linspace(-12, 12, num=24 + 1)
for i in tqdm(offset):
    tmp = (zenith(ds.julian_day - 1 + (ds.UTC + i)/24., ds.lat, ds.lon))*180./np.pi
    tmp = xr.where(tmp <= 90., tmp, 90.)
    tmp = np.cos(tmp*np.pi/180.)
    tmp = tmp.rename(str(i))
    if i == -12:
        solzc_mean = tmp
    else:
        solzc_mean = xr.merge([solzc_mean,tmp])

tmp = solzc_mean.to_array(dim='new').mean(dim='new')
scale = np.cos(solzc*np.pi/180.)/tmp

ds = ds.assign_coords(scale=("sample", 1./scale))

#compute daily mean SW heating and concatenate to the original dataset
SW_heat_dmean = ds.sel(var_names = 'SW_heat')*ds.scale
SW_heat_dmean = SW_heat_dmean.assign_coords(var_names = 'SW_heat_dmean')
ds_out = xr.concat([ds.arrvar, SW_heat_dmean.arrvar],"var_names")
ds_out = ds_out.to_dataset()

# Discard unrealistic data
print('Discarding unrealistic data...')
ds_out = ds_out.where(ds_out.sel(var_names = 'CTRC').arrvar > -1000., drop = True)
ds_out = ds_out.where(ds_out.sel(var_names = 'COD').arrvar > 0., drop = True)
ds_out = ds_out.where(ds_out.sel(var_names = 'COD').arrvar < 500., drop = True)
ds_out =  ds_out.where(ds_out.sel(var_names = 'LW_CRE') < 0., drop = True)
ds_out =  ds_out.where(ds_out.sel(var_names = 'SW_heat') > 0., drop = True)
ds_out =  ds_out.where(ds_out.scale < 1., drop = True)

#remove some useless variables to save space
ds_out = ds_out.drop(var_names=['solz', 'rh', 't2m', 'td'])

out_fn = "/lustre/ytzheng/Data/CTRC_Yannian/All_2014_NCEP_1stprcesd.nc"
ds_out.to_netcdf(out_fn)

print('Done!')
