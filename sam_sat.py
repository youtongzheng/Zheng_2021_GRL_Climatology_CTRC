import numpy as np
from sam_constants import a_esatw, a_dtesatw, a_esati, a_dtesati, eps

def esatw(t):
    """ Compute saturation vapor pressure over liquid water """
    dt = np.maximum(-80., t - 273.16)
    return np.polyval(a_esatw, dt)

def qsatw(t,p):
    """ Compute saturation specific humidity over liquid water """
    return esatw(t) / (eps * np.maximum(esatw(t), p - esatw(t)))

def esati(t):
    """ Compute saturation vapor pressure over ice """
    if not np.isscalar(t):
        dt = t - 273.16
        res = np.polyval(a_esati, dt)
        res[t >= 273.15] = esatw(t[t >= 273.15])
        dt = np.maximum(-100, t - 273.16)
        res[t <= 185] = 0.00763685 + dt[t <= 185]*(0.000151069 + dt[t <= 185]*7.48215e-07)
        return res
    elif t >= 273.15:
        return esatw(t)
    elif t <= 185:
        dt = t - 273.15
        return 0.00763685 + dt*(0.000151069 + dt*7.48215e-07)
    else:
        dt = t - 273.15
        return np.polyval(a_esati, dt)


def qsati(t,p):
    """ Compute saturation specific humidity over ice """
    return esati(t) / (eps * np.maximum(esati(t), p - esati(t)))

def dtesatw(t):
    """ 
    Compute temperature derivative of saturation vapor pressure
    over liquid water
    """
    dt = np.maximum(-80, t-273.16)
    return np.polyval(a_dtesatw, t)

def dtqsatw(t,p):
    """
    Compute temperature derivative of saturation specific humidity
    over liquid water
    """
    return dtesatw(t) / (eps * p)
