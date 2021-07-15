#!/usr/bin/env python

'''
Generate BAL catalogs from DESI coadd data for a specific data release. One 
catalog is generated per coadd file. The catalogs are put in a directory 
structure that matches the structure of the data release. 
Use the separate script buildbalcat.py to create a BAL catalog for a 
corresponding QSO catalog. 
'''

import os
import sys
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

debug = True

def pmmkdir(direct): 
    if not os.path.isdir(direct):
        try:
            os.makedirs(direct)
        except PermissionError:
            print("Error: no permission to make directory ", direct)
            exit(1)

os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Run balfinder on DESI data""")

parser.add_argument('-t','--tile', nargs='*', default = None, required=False,
                    help='List of tile number(s) to process - default is all')

parser.add_argument('-d', '--date', nargs='*', default = None, required = False,
                    help = 'List of observation dates to process - default is all')

parser.add_argument('-s', '--specprod', type = str, default = 'andes', required = False,
                    help = 'Specprod (data release subdirectory, default is andes)')

parser.add_argument('-o','--outdir', type = str, default = None, required = True,
                    help = 'Root directory for output files')

parser.add_argument('-c','--clobber', type=bool, default=False, required=False,
                    help='Clobber (overwrite) BAL catalog if it already exists?')

parser.add_argument('-v','--verbose', type = bool, default = False, required = False,
                    help = 'Provide verbose output?')


args  = parser.parse_args()

if debug: 
    args.verbose=True
    
release = args.specprod

# Root directory for input data: 
dataroot = os.path.join(os.getenv("DESI_SPECTRO_REDUX"), release, "tiles")
if release == 'daily':
    dataroot = os.path.join(dataroot, 'cumulative')
if not os.path.isdir(dataroot): 
    print("Error: did not find root directory ", dataroot)
    exit(1)
    
# Root directory for output catalogs: 
outroot = os.path.join(args.outdir, release, "tiles")
pmmkdir(outroot)

# List of tiles that caused issues for by hand rerun.
issuetiles = []

# Determine which tile(s) to process

tiledirs = glob(dataroot + "/*")

tiles = []  # list of all available tiles
for tiledir in tiledirs:
    tiles.append(tiledir[tiledir.rfind('/')+1::])

inputtiles = args.tile

if inputtiles is not None:
    for inputtile in inputtiles:
        assert(inputtile in tiles), "Tile {} not available".format(inputtile)
else:
    inputtiles = tiles

# Create/confirm output tile directories exist
for tile in inputtiles: 
    tiledir = os.path.join(outroot, tile) 
    pmmkdir(tiledir) 

# For each tile in inputtiles, get the list of dates, create output 
# directories, identify BALs, and create catalogs 

for tile in inputtiles: 
    # Identify dates
    tiledir = os.path.join(dataroot, tile) 
    datedirs = glob(tiledir + "/*")
    for datedir in datedirs: 
        date = datedir[datedir.rfind('/')+1::] 
        print(tile, date) 
        outdatedir = os.path.join(outroot, tile, date)
        pmmkdir(outdatedir) 
        coaddfiles = glob(datedir + "/coadd-*.fits")
        for coaddfile in coaddfiles: 
            coaddfilename = coaddfile[coaddfile.rfind('/')+1::]
            balfilename = coaddfilename.replace('coadd-', 'baltable-')
            if args.verbose: 
                print("Coadd file: ", coaddfile)

            if not os.path.isfile(balfilename) or args.clobber: 
                try:
                    db.desibalfinder(coaddfile, altbaldir=outdatedir, overwrite=args.clobber, verbose=True, release=release)
                except:
                    print("An error occured at tile {}. Adding tile to issuetiles list.".format(tile))
                    issuetiles.append(tile)
                    
print("Errors occured at the tiles: ")
for issuetile in issuetiles:
    print(issuetile, end=" "),