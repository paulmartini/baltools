#!/usr/bin/env python
"""

baltools.appendbalinfo_hp
=========================

Utilizes functinos from popqsotab.py to add empty BAL columns to existing
QSO catalogue and add information from baltables to new catalogue.
runbalfinder.py tables

2021 Original code by Simon Filbert

"""

import os
import sys
import numpy as np
from astropy.io import fits
import healpy as hp

import argparse

from baltools import balconfig as bc
from baltools import fitbal
from baltools import popqsotab as pt


def pmmkdir(direct): 
    if not os.path.isdir(direct):
        try:
            print(direct, "not found. Making new directory.")
            os.makedirs(direct)
        except PermissionError:
            print("Error: no permission to make directory ", direct)
            exit(1)

balcols = ['PCA_COEFFS', 'PCA_CHI2', 'BAL_PROB', 'BI_CIV', 'ERR_BI_CIV', 'NCIV_2000', 'VMIN_CIV_2000', 'VMAX_CIV_2000', 'POSMIN_CIV_2000', 'FMIN_CIV_2000', 'AI_CIV', 'ERR_AI_CIV', 'NCIV_450', 'VMIN_CIV_450', 'VMAX_CIV_450', 'POSMIN_CIV_450', 'FMIN_CIV_450', 'BI_SIIV', 'ERR_BI_SIIV', 'NSIIV_2000', 'VMIN_SIIV_2000', 'VMAX_SIIV_2000', 'POSMIN_SIIV_2000', 'FMIN_SIIV_2000', 'AI_SIIV', 'ERR_AI_SIIV', 'NSIIV_450', 'VMIN_SIIV_450', 'VMAX_SIIV_450', 'POSMIN_SIIV_450', 'FMIN_SIIV_450']

def balcopy(qinfo, binfo):
    for balcol in balcols: 
        qinfo[balcol] = binfo[balcol]
    qinfo['BALMASK'] = 0


os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Update existing QSO catalogue with BAL information""")

parser.add_argument('-q', '--qsocat', type = str, default = None, required = True,
                    help = 'Input QSO catalog')

parser.add_argument('-b','--baldir', type=str, default = None, required=True,
                    help='Path to directory structure with individual BAL catalogs')

parser.add_argument('-o','--outcat', type=str, default="qso-balcat.fits", 
                    required=False, help='Filename of output QSO+BAL catalog')

parser.add_argument('-s', '--survey', type = str, default = 'main', required = False,
                    help = 'Survey subdirectory [sv1, sv2, sv3, main], default is main')

parser.add_argument('-m', '--moon', type = str, default = 'dark', required = False,
                    help = 'Moon brightness [bright, dark], default is dark')

parser.add_argument('-c','--clobber', type=bool, default=False, required=False,
                    help='Clobber (overwrite) BAL catalog if it already exists?')

parser.add_argument('-v','--verbose', type = bool, default = False, required = False,
                    help = 'Provide verbose output?')


args  = parser.parse_args()

# Check the QSO catalog exists
if not os.path.isfile(args.qsocat):
    print("Error: cannot find ", args.qsocat)
    exit(1)
    
    
# Full path to the output QSO+BAL catalog
outcat = os.path.join(args.baldir, args.outcat) 

# Add empty BAL cols to qso cat and writes to outcat.
# Stores return value (BAL card names) in cols
cols = pt.inittab(args.qsocat, outcat)
# # Want to manually set this to -1 to show that it is not populated
# cols.remove('BAL_PROB')

qhdu = fits.open(outcat)
qcat = qhdu[1].data

# Calculate healpix for every QSO 
healpixels = hp.ang2pix(64, qcat['TARGET_RA'], qcat['TARGET_DEC'], lonlat=True, nest=True)

# Construct a list of unique healpix pixels
healpixlist = np.unique(healpixels)

# Construct an array of indices for the QSO catalog
hindxs = np.arange(0, len(qcat), dtype=int)

if args.verbose: 
    print("Found {0} entries with {1} unique healpix".format(len(healpixels), len(healpixlist)))

# /global/cfs/cdirs/desi/users/martini/everest/healpix/sv3/dark/93/9338/baltable-sv3-dark-9338.fits
for healpix in healpixlist: 
    balfilename = "baltable-{0}-{1}-{2}.fits".format(args.survey, args.moon, healpix) 
    balfile = os.path.join(args.baldir, str(healpix)[:len(str(healpix))-2], str(healpix), balfilename)
    bhdu = fits.open(balfile) 
    bcat = bhdu['BALCAT'].data
    hmask = healpixels == healpix
    if args.verbose: 
        print(healpix, len(str(healpix)), str(healpix)[:len(str(healpix))-2], balfile, os.path.isfile(balfile), np.sum(hmask))
    indxs = hindxs[hmask] # indices in qcat that are in pixel healpix
    for i, targetid in enumerate(bhdu['BALCAT'].data['TARGETID']):
        ind = np.where(targetid == qcat['TARGETID'])[0]
        if len(ind) > 0:
            print(i, targetid, qcat['TARGETID'][ind[0]], qcat['Z'][ind[0]], ind[0], qcat['BALMASK'][ind[0]], qcat['BI_CIV'][ind[0]])
            balcopy(qcat[ind[0]], bhdu['BALCAT'].data[i])
            print(i, targetid, qcat['TARGETID'][ind[0]], qcat['Z'][ind[0]], ind[0], qcat['BALMASK'][ind[0]], qcat['BI_CIV'][ind[0]])

qhdu.writeto(outcat, overwrite=True)
print("Wrote ", outcat) 

