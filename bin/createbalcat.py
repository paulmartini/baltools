#!/usr/bin/env python
  
'''
Create a BAL catalog based on a QSO catalog. This script assembles a BAL 
catalog for each QSO in the QSO catalog that is in the redshift range to 
identify BAL features. For each QSO it identifies the individual BAL 
catalog and takes the BAL information from there. 

Inputs must at least be the QSO catalog, data release / SPECPROD, and 
the root of the directory structure with the BAL catalogs. 

The individual BAL catalogs are created with runbalfinder.py
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack, hstack
from glob import glob

import argparse
from time import gmtime, strftime

import fitsio
import desispec.io
from desispec.coaddition import coadd_cameras

import baltools
from baltools import balconfig as bc
from baltools import plotter, fitbal, baltable
from baltools import desibal as db

debug = True

os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux/'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Assemble BAL catalog based on a QSO catalog""")

parser.add_argument('-q','--qsocatalog', type = str, default = None, required = True,
                    help = 'Input QSO catalog name, e.g. qsocat-minisv.fits')

parser.add_argument('-b','--baldir', type=str, default=".", required=False,
                    help='Path to directory structure with individual BAL catalogs')

parser.add_argument('-o','--outputcatalog', type=str, default="balcat.fits",
                    required=False, help='Output BAL catalog name')

parser.add_argument('-c','--clobber', type=bool, default=False, required=False,
                    help='Clobber (overwrite) BAL catalog if it already exists?')

parser.add_argument('-v','--verbose', type=bool, default=False, required=False,
                    help='Provide verbose output?')

args  = parser.parse_args()
balroot = args.baldir
qsocatname = args.qsocatalog
balcatname = args.outputcatalog

if debug: 
    args.clobber = True
    args.verbose = True

try: 
    qtab = Table.read(qsocatname)
except FileNotFoundError:
    print("Error: ", qsocatname, "not found")
    exit(1)

release = fits.getval(qsocatname, 'RELEASE', ext=1)
dataroot = os.path.join(os.getenv("DESI_SPECTRO_REDUX"), release, "tiles")

if not os.path.exists(balroot):
    print("Error: root BAL directory does not exist: ", balroot)

if not os.path.exists(dataroot):
    print("Error: root data directory does not exist: ", dataroot)

if args.verbose:
    print("Root data directory = ", dataroot)
    print("Root BAL directory = ", balroot)
    print("Output catalog name = ", balcatname)

# For each QSO in the catalog: 
# - check if in BAL redshift range
# - identify tileid, date, spectro
# - open that baltable
# - add to BAL catalog

first = True

for i in range(len(qtab)):
    if i == 0: 
        # Create empty BAL row for non-BALs based on a known, non-BAL
        tileid = '68002'
        night = '20200315'
        sp = '7'
        baltablename = os.path.join(balroot, tileid, night, "baltable-"+sp+"-"+tileid+"-"+night+".fits")
        targetid = 35185977857147541# Known non-BAL
        baltab = Table.read(baltablename)
        bindx = np.where(baltab['TARGETID'] == targetid)[0][0]
        emptybal = baltab.copy()
        emptybal.remove_columns(['TARGETID', 'TARGET_RA', 'TARGET_DEC',
            'Z', 'ZERR', 'ZWARN', 'NIGHT', 'EXPID', 'MJD', 'TILEID'])
        emptybal = emptybal[bindx]
        emptybal['BAL_PROB'] = -1.
    zspec = qtab['Z'][i]
    targetid = qtab['TARGETID'][i]
    print(i, "TARGETID, z = ", targetid, zspec)
    if zspec >= bc.BAL_ZMIN and zspec <= bc.BAL_ZMAX: 
        tileid = str(qtab['TILEID'][i])
        night = str(qtab['NIGHT'][i])
        sp = str(qtab['PETAL_LOC'][i])
        baltablename = os.path.join(balroot, tileid, night, "baltable-"+sp+"-"+tileid+"-"+night+".fits")
        if not os.path.isfile(baltablename):
            print("Warning: BAL catalog for TARGETID = ", targetid, "not found: ", baltablename)
        baltab = Table.read(baltablename)
        bindx = np.where(baltab['TARGETID'] == targetid)[0][0]
        baltab.remove_columns(['TARGETID', 'TARGET_RA', 'TARGET_DEC', 
            'Z', 'ZERR', 'ZWARN', 'NIGHT', 'EXPID', 'MJD', 'TILEID'])
        newrow = hstack([qtab[i], baltab[bindx]])
        if first:
            outtab = newrow
            first = False
        else: 
            outtab = vstack([outtab, newrow])
    else:
        newrow = hstack([qtab[i], emptybal])
        if first:
            outtab = newrow
            first = False
        else: 
            outtab = vstack([outtab, newrow])


outtab.write(balcatname, format='fits', overwrite=True)
lastupdate = "Last updated {0} UT by {1}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), os.getlogin())
commandline = " ".join(sys.argv)
fits.setval(balcatname, 'RELEASE', value=release, ext=1)
fits.setval(balcatname, 'COMMENT', value=commandline, ext=1)
fits.setval(balcatname, 'HISTORY', value=lastupdate, ext=1)
if args.verbose:
    print("Wrote output BAL catalog ", balcatname)
