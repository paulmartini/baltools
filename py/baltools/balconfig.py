"""

baltools.balconfig
==================

Common modules and global variables used by balfinder

"""

from __future__ import print_function, absolute_import, division
import os
import socket
from astropy import constants as const
import fitsio 

c = const.c.to('km/s').value	     # speed of light in km/s

# Wavelengths of BAL features (approximate)
lambdaCIV = 1549.
lambdaSiIV = 1398.

# Velocity search range for BAL troughs
VMIN_BAL = -25000.  # minimum search velocity
VMAX_BAL = 0.

BAL_LAMBDA_MIN = 1261.    # Minimim wavelength for PCA
BAL_LAMBDA_MAX = 2399.    # Maximum wavelength for PCA
# BAL_LAMBDA_MAX = 1800.    # Maximum wavelength for PCA

# Redshift range for BAL Catalog
# QSOs only within BAL_ZMIN <= z <= BAL_ZMAX
BAL_ZMIN = 1.57
BAL_ZMAX = 5.0 # Use this for DR14 
# BAL_ZMAX = 5.0 # Changed Mar 2020 by PM

NPCA = 5	    # Number of PCA coefficients
NBI = 5		# Max number of potential BI troughs
NAI = 17	    # Max number of potential AI troughs

homedir = os.environ['HOME']
hostname = socket.gethostname()

# if 'cori' in hostname: 
if '/global/homes' in homedir:  # should be true if at NERSC
    catdir = homedir + '/Catalogs/'
    specdir1 = '/global/projecta/projectdirs/sdss/staging/dr14/eboss/spectro/redux/v5_10_0/spectra/lite/'
    specdir2 = '/global/projecta/projectdirs/sdss/staging/dr9/sdss/spectro/redux/'
    # specdir = '/global/projecta/projectdirs/sdss/staging/dr14/sdss/spectro/redux/v5_10_0/spectra/lite/' 
elif 'Users' in homedir: 
    catdir = homedir + '/Catalogs/'
    specdir = homedir + '/Data/'
elif os.getlogin() == 'u6024124':
    catdir = homedir + '/Catalogs/'
    specdir = '/uufs/chpc.utah.edu/common/home/sdss/ebosswork/eboss/spectro/redux/v5_13_0/spectra/lite/' 
    # specdir = '/uufs/chpc.utah.edu/common/home/sdss/dr16/sdss/spectro/redux/26/'
else: 
    specdir = 'Data/'
    catdir = 'Catalogs/'

baldr12file = catdir + 'DR12Q_BAL.fits'
qsodr12file = catdir + 'DR12Q.fits'
qsodr14file = catdir + 'DR14Q_v4_4.fits'
pcaeigenfile = catdir + 'PCA_Eigenvectors.fits'
baldr14file = catdir + 'DR14Q_BAL_v_2_0.fits'
qsodr16file = catdir + 'DR16Q_QSOCat_20190703.fits'
desispecdir1 = '/global/cscratch1/sd/mjwilson/svdc2019c/spectro/redux/v1/spectra-64/'
desispecdir2 = '/project/projectdirs/desi/datachallenge/reference_runs/19.2/spectro/redux/mini/spectra-64/'
desispecdir3 = '/project/projectdirs/desi/mocks/lya_forest/develop/london/v5.0.0/quick-2.6/spectra-16/'

# Check the wavelength limits for the PCA fit are consistent 
# with the templates

pcaeigen = fitsio.read(pcaeigenfile)

try: 
    assert (BAL_LAMBDA_MIN >= pcaeigen['WAVE'][0])
except AssertionError:
    print("Error: BAL_LAMBDA_MIN {0:.0f} must be greater than or equal to {1:.2f} in {2}".format(BAL_LAMBDA_MIN, pcaeigen['WAVE'][0], pcaeigenfile))
  
try: 
    assert (BAL_LAMBDA_MAX <= pcaeigen['WAVE'][-1])
except AssertionError:
    print("Error: BAL_LAMBDA_MAX {0:.0f} must be less than or equal to {1:.2f} in {2}".format(BAL_LAMBDA_MAX, pcaeigen['WAVE'][-1], pcaeigenfile)) 

