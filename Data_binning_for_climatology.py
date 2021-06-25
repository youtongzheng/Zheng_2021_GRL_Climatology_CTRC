import numpy as np
import xarray as xr
import datetime
from tqdm import tqdm
import sys

# Parse inputs
if len(sys.argv) !=4:
    print('ERROR---Usage: python Postprc_SAM.py ' +
        '<dir> <input_file_name (without extension)> <output_file_name (without extension)>')
    exit()

file_input = sys.argv[1] + sys.argv[2] 
file_output = sys.argv[3]

# ds  = xr.open_dataset("/lustre/ytzheng/Data/CTRC_Yannian/" + sat + "2014_NCEP_1stprcesd.nc")
ds  = xr.open_dataset(file_input + '.nc')

print('Finish reading the postprocessed data, start binning...')

#Compute global annual mean
print('Computing global annual mean...')

lon_out = np.linspace(-179, 179, num=359)
var_names = ds.var_names.values
nvar = len(var_names)

# a = ds.groupby(ds.lat0)
# lat_out = []
# for indx, (label, group) in enumerate(tqdm(a)):
#     tmp_mean = np.full((1, 359, nvar), -999.)
#     tmp_std = np.full((1, 359, nvar), -999.)
#     tmp_count = np.full((1, 359, nvar), -999.)
    
#     ind = np.intersect1d(lon_out, group.groupby(group.lon0).mean().lon0.values, return_indices = True)    
#     tmp_mean[0, ind[1],:] = group.groupby(group.lon0).mean().arrvar.values 
#     tmp_std[0, ind[1],:] = group.groupby(group.lon0).std().arrvar.values
#     tmp_count[0, ind[1],:] = group.groupby(group.lon0).count().arrvar.values
    
#     if indx == 0:
#         tmp_mean_out = tmp_mean
#         tmp_std_out = tmp_std
#         tmp_count_out = tmp_count
#     else:
#         tmp_mean_out = np.concatenate((tmp_mean_out, tmp_mean), axis = 0)
#         tmp_std_out = np.concatenate((tmp_std_out, tmp_std), axis = 0)
#         tmp_count_out = np.concatenate((tmp_count_out, tmp_count), axis = 0)
        
#     lat_out.append(label)
    
# for ivar, var in enumerate(var_names):
#     ds_tmp = xr.DataArray(np.squeeze(tmp_mean_out[:,:,ivar]), coords=[lat_out , lon_out],
#                        dims=["Latitude", "Longitude"]).rename(var)
    
#     if ivar == 0:
#         ds_out = ds_tmp
#     else:
#         ds_out = xr.merge([ds_out, ds_tmp]) 

# for ivar, var in enumerate(var_names):
#     ds_tmp = xr.DataArray(np.squeeze(tmp_std_out[:,:,ivar]), coords=[lat_out , lon_out],
#                        dims=["Latitude", "Longitude"]).rename(var + '_std')

#     ds_out = xr.merge([ds_out, ds_tmp]) 

# ds_tmp = xr.DataArray(np.squeeze(tmp_count_out[:,:,ivar]), coords=[lat_out , lon_out],
#                       dims=["Latitude", "Longitude"]).rename('count')
# ds_out = xr.merge([ds_out, ds_tmp]) 

# ds_out.to_netcdf(file_output + "_gridded.nc")

#Compute global seasonal mean
print('Computing global seasonal mean...')

season_arr = ['DJF','MAM','JJA','SON']
for season in season_arr:
    ds0 = ds.where(ds.time.dt.season == season, drop = True)
    a = ds0.groupby(ds0.lat0)

    lat_out = []
    for indx, (label, group) in enumerate(tqdm(a)):
        tmp_mean = np.full((1, 359, nvar), -999)
        tmp_std = np.full((1, 359, nvar), -999)
        tmp_count = np.full((1, 359, nvar), -999)

        ind = np.intersect1d(lon_out, group.groupby(group.lon0).mean().lon0.values, return_indices = True)    
        tmp_mean[0, ind[1],:] = group.groupby(group.lon0).mean().arrvar.values 
        tmp_std[0, ind[1],:] = group.groupby(group.lon0).std().arrvar.values
        tmp_count[0, ind[1],:] = group.groupby(group.lon0).count().arrvar.values

        if indx == 0:
            tmp_mean_out = tmp_mean
            tmp_std_out = tmp_std
            tmp_count_out = tmp_count
        else:
            tmp_mean_out = np.concatenate((tmp_mean_out, tmp_mean), axis = 0)
            tmp_std_out = np.concatenate((tmp_std_out, tmp_std), axis = 0)
            tmp_count_out = np.concatenate((tmp_count_out, tmp_count), axis = 0)

        lat_out.append(label)

    # lat_out = np.linspace(int(list(a)[0][0]), int(list(a)[-1][0]), num=int(list(a)[-1][0]-list(a)[0][0]+1))

    for ivar, var in enumerate(var_names):
        ds_tmp = xr.DataArray(np.squeeze(tmp_mean_out[:,:,ivar]), coords=[lat_out , lon_out],
                           dims=["Latitude", "Longitude"]).rename(var)

        if ivar == 0:
            ds_out = ds_tmp
        else:
            ds_out = xr.merge([ds_out, ds_tmp]) 

    for ivar, var in enumerate(var_names):
        ds_tmp = xr.DataArray(np.squeeze(tmp_std_out[:,:,ivar]), coords=[lat_out , lon_out],
                           dims=["Latitude", "Longitude"]).rename(var + '_std')

        ds_out = xr.merge([ds_out, ds_tmp]) 

    ds_tmp = xr.DataArray(np.squeeze(tmp_count_out[:,:,ivar]), coords=[lat_out , lon_out],
                          dims=["Latitude", "Longitude"]).rename('count')
    ds_out = xr.merge([ds_out, ds_tmp]) 
    ds_out.to_netcdf(file_output + "_gridded_" + season + ".nc")
    

#Compute global seasonal mean
print('Computing regional seasonal mean...')
from Regions import *

for lct in region:
    tmp = ds.where(ds.lat > region[lct][0]
                 ).where(ds.lat < region[lct][1]
                        ).where(ds.lon > region[lct][2]
                               ).where(ds.lon < region[lct][3]).groupby("time.season").mean().arrvar.rename(lct)
    if lct == 'NEP':
        ds_season = tmp
    else:
        ds_season = xr.merge([ds_season, tmp])
        
ds_season = xr.concat([ds_season.sel(season = 'DJF'), 
                           ds_season.sel(season = 'MAM'),
                           ds_season.sel(season = 'JJA'),
                           ds_season.sel(season = 'SON')],"season")

ds_season.to_netcdf(file_output + "_season_averaged.nc")

print('Done!')
