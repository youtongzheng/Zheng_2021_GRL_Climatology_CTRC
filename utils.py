import matplotlib.pyplot as plt
from matplotlib import cm
import xarray as xr
from sam_sat import *

def plt2D(fig, ax, x, y, z, ztitle, mycmap=cm.Spectral_r, mylevels=[0.], extend = 'both', myfontsize = 11, 
         yrange = [-65., 65.], xrange = [-180, 180], cbar = True, xlabel = False, ylabel = False, out = False):
    if len(mylevels) == 1:
        oax = ax.contourf(x,y,z, cmap = mycmap, extend = extend)
    else:
        oax = ax.contourf(x,y,z, cmap = mycmap, levels = mylevels, extend = extend)
    
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    
    ax.set_yticks(ticks=[-60, -30, 0, 30, 60])
    
    if ylabel:
        ax.set_ylabel('Latitude ($\degree$)', fontsize=myfontsize) 
    
    if xlabel:
        ax.set_xlabel('Longitude ($\degree$)', fontsize=myfontsize) 
        
    ax.grid(linestyle='--', linewidth=0.3)
    ax.set_title(ztitle, fontsize=0.8*myfontsize)
    
    if cbar:
        cbar = fig.colorbar(oax,  ax=ax, orientation='vertical', shrink=0.8, pad = 0.04,
                            ticks = [mylevels.min(), mylevels.max()])
        
#         cbar.set_label(ztitle, fontsize=0.8*myfontsize)
    if out:
        return oax

def getsnd(f, lct):
    name_season = np.array(['DJF','MAM','JJA','SON'])
    
    tmp = f[lct].isel(var_names = slice(11, 31, 2)).values
    rh = f[lct].isel(var_names = slice(12, 31, 2)).values
    pres = np.linspace(100, 1000, num=10)
    q = 1000.*qsatw(tmp,pres)*rh/100.

    ds_rh = xr.DataArray(rh, coords=[name_season , pres],
                           dims=["season", "pressure"]).rename('rh')
    ds_tmp = xr.DataArray(tmp, coords=[name_season , pres],
                           dims=["season", "pressure"]).rename('tmp')
    ds_q = xr.DataArray(q, coords=[name_season , pres],
                           dims=["season", "pressure"]).rename('q')

    ctp = []
    ctq = []
    ctt = []
    for var in name_season:
        tmp_tmp = ds_tmp.sel(season = var).values
        lapse = (tmp_tmp[-1] - tmp_tmp[-2])/100.
        tmp_tmp0 = tmp_tmp[-1] + lapse*(pres - 1000.)
        
        q_tmp = ds_q.sel(season = var).values

        ctt_tmp = f[lct].sel(var_names = 'CTT').sel(season = var).values + 273.15
        ctt.append(ctt_tmp)

        ctp_tmp = np.interp(ctt_tmp, tmp_tmp0, pres)
        ctp.append(ctp_tmp)

        ctq.append(np.interp(ctp_tmp, pres, q_tmp))

    ds_ctp = xr.DataArray(ctp, coords=[name_season],
                           dims=["season"]).rename('ctp')
    ds_ctq = xr.DataArray(ctq, coords=[name_season],
                           dims=["season"]).rename('ctq')
    ds_ctt = xr.DataArray(ctt, coords=[name_season],
                           dims=["season"]).rename('ctt')
    ds = xr.merge([ds_rh, ds_tmp, ds_q, ds_ctp, ds_ctq,ds_ctt])
    
    return ds

def pltsnd(fig, ax, ds, title, legend = False,xlabel = False, ylabel = False, notick = True, myfontsize = 11):
    name_season = np.array(['DJF','JJA'])
    for var in name_season:
        a = ax.plot(ds['q'].sel(season = var), ds.pressure, '-', label = var)
        ax.plot(ds['ctq'].sel(season = var), ds['ctp'].sel(season = var), marker = '+', color = a[0].get_color())

    ax.set_ylim([300., 1000])
    ax.set_xlim([-0.5, 14])
    ax.invert_yaxis()
    ax.set_title(title)
    
    if legend:
        ax.legend(loc="best", fontsize=0.8*myfontsize)
    
    if ylabel:
        ax.set_ylabel('Pressure (hPa)', fontsize=0.8*myfontsize) 
    
    if xlabel:
        ax.set_xlabel('q ($gkg^{-1}$)', fontsize=0.8*myfontsize) 
    
    if notick:
        ax.axes.yaxis.set_ticklabels([])

# ;+
# ; NAME:
# ;	zenith
# ;
# ; PURPOSE:
# ;	Calculate the solar zenith angle (and optionally the omega angle (see below))
# ;	
# ; CATEGORY:
# ;	FUNCTION
# ;
# ; CALLING SEQUENCE:
# ;	zenith(utc,lat,lon)
# ;
# ; EXAMPLE:
# ;	zenith(201.35,20,10.)
# ;
# ; INPUTS: 
# ;	utc	flt or fltarr: the time (UTC) as days since Jan 1 00:00 of the year
# ;		e.g. the time 0.5 is noon of Jan 1st.
# ;		(presently it also works for the following year, if the first year
# ;		 had 365 days. I.e it works for the NOXAR times for 1995 and 1996 but
# ;		 not yet for 1997
# ;	lat	flt or fltarr: the latitude in deg (pos. for northern hemisphere)
# ;	lon	flt or fltarr: the longitude in deg (pos. for eastern longitudes)
# ;
# ; OPTIONAL INPUT PARAMETERS:
# ;
# ; KEYWORD INPUT PARAMETERS:
# ;
# ; OUTPUTS
# ;	the solar zenith angle in radian
# ;	omega	: the hour angle in radian
# ;		  the local time (in fractional hours) can be approximated by
# ;		  calculating 12+omega*180/!pi*24/360.
# ; COMMON BLOCKS:
# ;
# ; SIDE EFFECTS: 
# ;
# ; RESTRICTIONS:
# ;	
# ; PROCEDURE:
# ;	
# ; MODIFICATION HISTORY:
# ;	first implementation Jan, 17 1997 by Dominik Brunner
# ;	adapted from the JNO2 program by Wiel Wauben, KNMI
# ;-
def zenith(utc, lat, lon):
#     calculate the number of days since the 1.1. of the specific year
    daynum=np.floor(utc)+1

#     calculate the relative SUN-earth distance for the given day
#     resulting from the elliptic orbit of the earth
    eta=2.*np.pi*daynum/365.

    fluxfac = 1.000110 + 0.034221*np.cos(eta) + 0.000719*np.cos(2.*eta) + 0.001280*np.sin(eta) + 0.000077*np.sin(2.*eta)
    dist = 1./np.sqrt(fluxfac)
    
#      calculate the solar declination for the given day
#      the declination varies due to the fact, that the earth rotation axis
#      is not perpendicular to the ecliptic plane
    delta = 0.006918 - 0.399912*np.cos(eta)-0.006758*np.cos(2.*eta)-0.002697*np.cos(3.*eta) + 0.070257*np.sin(eta)+0.000907*np.sin(2.*eta)+0.001480*np.sin(3.*eta)
    
#      equation of time, used to compensate for the earth's elliptical orbit
#      around the sun and its axial tilt when calculating solar time
#      eqt is the correction in hours
    et=2.*np.pi*daynum/366.
    eqt= 0.0072*np.cos(et)-0.0528*np.cos(2.*et) - 0.0012*np.cos(3.*et) - 0.1229*np.sin(et) - 0.1565*np.sin(2.*et) - 0.0041*np.sin(3.*et)
    
#      calculate the solar zenith angle
    dtr=np.pi/180. # degrees to radian conversion factor
    time = (utc + 1.-daynum)*24 # time in hours
    omega = (360./24.)*(time + lon/15. + eqt - 12.)*dtr
    sinh = np.sin(delta)*np.sin(lat*dtr) + np.cos(delta)*np.cos(lat*dtr)*np.cos(omega)
    solel = np.arcsin(sinh)
    sza = np.pi/2. - solel

    return sza