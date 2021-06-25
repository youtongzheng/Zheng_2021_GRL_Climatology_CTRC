import numpy as np

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