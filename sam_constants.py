import numpy as np

cp = 1004.
""" Specific heat of dry air (J kg-1 K-1) """
grav = 9.81
""" Gravitational acceleration (m s-2) """
lcond = 2.5104e6
""" Latent heat of condensation (J kg-1) """
lfus = 0.3336e6
""" Latent heat of fusion (J kg-1) """
lsub = lcond + lfus
""" Latent heat of sublimation (J kg-1) """
rv = 461.
""" Water vapor specific gas constant (J kg-1 K-1) """
rgas = 287.
""" Dry air specific gas constant (J kg-1 K-1) """
diffelq = 2.21e-5
""" Water vapor diffusivity (m2 s-1) """
therco = 2.40e-2
""" Air thermal conductivity (???) """
muelq = 1.717e-5
""" Dynamic viscosity of air (???) """
fac_cond = lcond/cp
""" Temperature of condensation (K) """
fac_fus = lfus/cp
""" Temperature of fusion (K) """
fac_sub = lsub/cp
""" Temperature of sublimation (K) """
eps = rv/rgas
""" Molecular mass ratios (dry air/water vapor) """

a_esatw = np.flip(np.array([
    6.11239921, 0.443987641, 0.142986287e-1,
    0.264847430e-3, 0.302950461e-5, 0.206739458e-7,
    0.640689451e-10, -0.952447341e-13,-0.976195544e-15]))
""" Coefficients for polynomial fit to SVP over water """
a_dtesatw = np.flip(np.array([
    0.443956472, 0.285976452e-1, 0.794747212e-3,
    0.121167162e-4, 0.103167413e-6, 0.385208005e-9,
    -0.604119582e-12, -0.792933209e-14, -0.599634321e-17]))
""" Coefficients for polynomial fit to T derivative of SVP over water """
a_esati = np.flip(np.array([
    6.11147274, 0.503160820, 0.188439774e-1,
    0.420895665e-3, 0.615021634e-5,0.602588177e-7,
    0.385852041e-9, 0.146898966e-11, 0.252751365e-14]))
""" Coefficients for polynomial fit to SVP over ice """
a_dtesati = np.flip(np.array([
    0.503223089, 0.377174432e-1,0.126710138e-2,
    0.249065913e-4, 0.312668753e-6, 0.255653718e-8,
    0.132073448e-10, 0.390204672e-13, 0.497275778e-16]))
""" Coefficients for polynomial fit to T derivative of SVP over ice """
