# baltools.balconfig
# ==================
#
# Common modules and global variables used by balfinder

import os
import socket

import fitsio
from astropy import constants as const

c = const.c.to('km/s').value	     # speed of light in km/s

# Wavelengths of BAL features (approximate)
lambdaCIV = 1549.
lambdaSiIV = 1398.

# --- Parameters for BAL Identification ---

# Velocity search range for BAL troughs
VMIN_BAL = -25000.  # minimum search velocity (km/s)
VMAX_BAL = 0.       # maximum search velocity (km/s)

# Velocity limits for specific indices
AI_VMAX = 0.        # max velocity for AI trough search (km/s)
BI_VMAX = -3000.    # max velocity for BI trough search (km/s)

# Trough definition parameters
AI_MIN_WIDTH = 450.         # min width for an AI trough (km/s)
BI_MIN_WIDTH = 2000.        # min width for a BI trough (km/s)
CONTINUUM_THRESHOLD = 0.9   # Flux must be below this fraction of the continuum
ERROR_SCALING_FACTOR = 0.5  # Factor to scale error term in trough identification

# --- PCA Fit Configuration ---

BAL_LAMBDA_MIN = 1261.    # Minimim wavelength for PCA fit
BAL_LAMBDA_MAX = 2399.    # Maximum wavelength for PCA fit

# Redshift range for BAL Catalog
BAL_ZMIN = 1.57
BAL_ZMAX = 5.0

NPCA = 5	    # Number of PCA coefficients
NBI = 5		    # Max number of potential BI troughs
NAI = 17	    # Max number of potential AI troughs

# --- File Paths ---
# (Path logic remains the same)

homedir = os.environ['HOME']
hostname = socket.gethostname()

if '/global/homes' in homedir:  # NERSC environment
    catdir = os.path.join(homedir, 'Catalogs/')
    specdir1 = '/global/cfs/projectdirs/sdss/staging/dr14/eboss/spectro/redux/v5_10_0/spectra/lite/'
    specdir2 = '/global/cfs/projectdirs/sdss/staging/dr9/sdss/spectro/redux/'
elif 'Users' in homedir:  # macOS environment
    catdir = os.path.join(homedir, 'Catalogs/')
    specdir = os.path.join(homedir, 'Data/')
elif os.getlogin() == 'u6024124': # CHPC environment
    catdir = os.path.join(homedir, 'Catalogs/')
    specdir = '/uufs/chpc.utah.edu/common/home/sdss/ebosswork/eboss/spectro/redux/v5_13_0/spectra/lite/'
else:
    specdir = 'Data/'
    catdir = 'Catalogs/'

pcaeigenfile = os.path.join(catdir, 'PCA_Eigenvectors.fits')
# Other file paths...
qsodr14file = os.path.join(catdir, 'DR14Q_v4_4.fits')
baldr14file = os.path.join(catdir, 'DR14Q_BAL_v_2_0.fits')


# --- Sanity Check PCA Wavelength Limits ---

try:
    pcaeigen = fitsio.read(pcaeigenfile)
    try:
        assert (BAL_LAMBDA_MIN >= pcaeigen['WAVE'][0])
    except AssertionError:
        print(f"Error: BAL_LAMBDA_MIN {BAL_LAMBDA_MIN:.0f} must be >= {pcaeigen['WAVE'][0]:.2f} in {pcaeigenfile}")

    try:
        assert (BAL_LAMBDA_MAX <= pcaeigen['WAVE'][-1])
    except AssertionError:
        print(f"Error: BAL_LAMBDA_MAX {BAL_LAMBDA_MAX:.0f} must be <= {pcaeigen['WAVE'][-1]:.2f} in {pcaeigenfile}")
except (FileNotFoundError, OSError):
    print(f"Warning: Could not open PCA eigenvector file: {pcaeigenfile}")

