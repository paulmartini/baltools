#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob

import argparse

import fitsio
from collections import defaultdict
import desispec.io
from desispec.coaddition import coadd_cameras

import baltools
from baltools import balconfig as bc
from baltools import plotter, fitbal, baltable
from baltools import desibal as db

os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux'
os.environ['SPECPROD'] = 'daily'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Run balfinder on DESI data""")

parser.add_argument('-t','--tile', type = int, default = None, required = True,
                    help = 'Tile number') 

parser.add_argument('-d', '--date', type = int, default = None, required = False,
                    help = 'Specify observation dates to process or do all by default')

parser.add_argument('-s', '--spectros', type = str, default = None, required = False,
                    help = 'Spectrographs to process (e.g. -s=3,5) or all by default')

parser.add_argument('-o','--outdir', type = str, default = None, required = False,
                    help = 'Path for output BAL catalog(s)')

parser.add_argument('-r','--redo', type = bool, default = False, required = False,
                    help = 'Redo (overwrite) BAL catalog if it already exists?')

parser.add_argument('-v','--verbose', type = bool, default = False, required = False,
                    help = 'Provide verbose output?')

args  = parser.parse_args()

# Identify the tile: 
tile = str(args.tile)

# Identify the observation date(s) 
if args.date is not None: 
    dates = args.date
else: 
    tiledir = os.path.join(os.getenv("DESI_SPECTRO_REDUX"), os.getenv("SPECPROD"), "tiles", tile) 
    datedirs = glob(tiledir + "/*")
    dates = []
    for path in datedirs:
        dates.append(path[path.rfind('/')+1::])

# Set (and create if necessary) the root output directory
# Note: add error checking on directory creation

if args.outdir is None: 
    rootbaldir = os.path.join('/global/cfs/cdirs/desi/users/', os.getlogin(), tile) 
else: 
    rootbaldir = args.outdir

if not os.path.exists(rootbaldir): 
    os.makedirs(rootbaldir) 

# Create output directories for each date

outbaldirs = [] 
for date in dates: 
    outbaldir = os.path.join(rootbaldir, date)
    if not os.path.exists(outbaldir): 
        os.makedirs(outbaldir) 
    outbaldirs.append(outbaldir)

# Determine which spectrographs to process (default it all) 

if args.spectros is None: 
    spectros = np.arange(0, 10) 
else: 
    spectros = [int(i) for i in args.spectros.split(',')]

if args.verbose: 
    print("Tile = ", tile) 
    print("Date(s) = ", dates) 
    print("Root output directory = ", rootbaldir)
    print("Output directories = ", outbaldirs)
    print("Spectrographs = ", spectros)

for i in range(len(dates)): 
    date = dates[i]
    for spectro in spectros: 
        spectrograph = str(spectro)
        dirname = os.path.join(os.getenv("DESI_SPECTRO_REDUX"), os.getenv("SPECPROD"), "tiles", tile, date)
        filename = "coadd-{}-{}-{}.fits".format(spectrograph, tile, date)
        specfilename = os.path.join(dirname, filename)
        if args.verbose: 
            print("Spec file name:", specfilename)
  
        # Filename for BAL catalog
        outbaldir = outbaldirs[i]
        baltmp = specfilename.replace('coadd-', 'baltable-')
        balfilename = os.path.join(outbaldir + "/", baltmp[baltmp.rfind("baltable-")::])
        if args.verbose: 
            print("BAL catalog name:", balfilename)

        if not os.path.isfile(balfilename) or args.redo: 
            db.desibalfinder(specfilename, altbaldir=outbaldir, overwrite=args.redo, verbose=True)

