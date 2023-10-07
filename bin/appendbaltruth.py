#!/usr/bin/env python
"""

baltools.appendbaltruth
=======================

For mocks.

Append the truth catalog information to the quasar catalog to create a catalog 
more similar to the observed BAL catalog. 

"""

import os
import sys
import numpy as np
from astropy.io import fits
import healpy as hp

import argparse
from time import gmtime, strftime

from baltools import balconfig as bc
from baltools import popqsotab as pt
from baltools import utils

# Columns in mock truth BAL catalog
truthcols = ['BAL_PROB', 'BI_CIV', 'ERR_BI_CIV', 'NCIV_2000', 'VMIN_CIV_2000', 'VMAX_CIV_2000', 'POSMIN_CIV_2000', 'FMIN_CIV_2000', 'AI_CIV', 'ERR_AI_CIV', 'NCIV_450', 'VMIN_CIV_450', 'VMAX_CIV_450', 'POSMIN_CIV_450', 'FMIN_CIV_450']

def balcopy(qinfo, binfo):
    for balcol in truthcols: 
        try: 
            qinfo[balcol] = binfo[balcol]
        except: 
            continue
    qinfo['BALMASK'] = 0


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Update existing QSO catalogue with BAL information""")

parser.add_argument('-q', '--qsocat', type = str, default = None, required = True,
                    help = 'Input QSO catalog (e.g. zcat.fits)')

parser.add_argument('-b','--balcat', type=str, default = None, required=True,
                    help='Input BAL truth catalog (e.g. bal_cat.fits)')

parser.add_argument('-o','--outcat', type=str, default=None, 
                    required=False, help='Output QSO+BAL catalog name (default appends -baltrue.fits)')

parser.add_argument('-c','--clobber', default=False, required=False, action='store_true', 
                    help='Clobber (overwrite) BAL catalog if it already exists?')

parser.add_argument('-v','--verbose', default=False, required=False, action='store_true', 
                    help = 'Provide verbose output?')

args  = parser.parse_args()

# Check the QSO catalog exists
if not os.path.isfile(args.qsocat):
    print("Error: cannot find ", args.qsocat)
    exit(1)
    
# Check the BAL truth catalog exists
if not os.path.isfile(args.balcat):
    print("Error: cannot find ", args.balcat)
    exit(1)

if args.outcat is None:
    outcat = args.qsocat[:args.qsocat.rfind('.fits')] + '-baltrue.fits'
else: 
    outcat = args.outcat

# Check if the output catalog exists 
if os.path.isfile(outcat):
    if not args.clobber: 
        print(f"Error: {outcat} exists and clobber=False")
        exit(1)
    
# Initialize the new catalog with the BAL columns
cols = pt.inittab(args.qsocat, outcat, alttemp=False) 

qhdu = fits.open(outcat)

bhdu = fits.open(args.balcat) 

# Loop through each QSO
for qindx,targid in enumerate(qhdu['ZCATALOG'].data['TARGETID']): 
    bindx = np.where(targid == bhdu[1].data['TARGETID'])[0]
    if len(bindx) > 0: 
        bindx = bindx[0]
        balcopy(qhdu['ZCATALOG'].data[qindx], bhdu['ZCATALOG'].data[bindx])
        print(targid, qhdu[1].data['TARGETID'][qindx], bhdu[1].data['TARGETID'][bindx], qhdu[1].data['AI_CIV'][qindx])

qhdu[1].header['EXTNAME'] = 'ZCATALOG'
qhdu.writeto(outcat, overwrite=True)
print("Wrote ", outcat) 
