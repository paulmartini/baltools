#!/usr/bin/env python

'''
Generate BAL catalogs from DESI healpix data for a specific data release. One 
catalog is generated per healpix. The catalogs are put in a directory 
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

from time import gmtime, strftime
import argparse
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

parser.add_argument('-hp','--healpix', nargs='*', default = None, required=False,
                    help='List of healpix number(s) to process - default is all')

parser.add_argument('-r', '--release', type = str, default = 'everest', required = False,
                    help = 'Data release subdirectory, default is everest')

parser.add_argument('-a','--altzdir', type=str, default = None, required=True,
                    help='Path to directory structure with healpix-based afterburner redshift catalogs')

parser.add_argument('-z','--zfileroot', type=str, default= None, required=False, 
                    help='Root name of healpix-based afterburner redshift catalogs')

parser.add_argument('-s', '--survey', type = str, default = 'main', required = False,
                    help = 'Survey subdirectory [sv1, sv2, sv3, main], default is main')

parser.add_argument('-m', '--moon', type = str, default = 'dark', required = False,
                    help = 'Moon brightness [bright, dark], default is dark')

parser.add_argument('-o','--outdir', type = str, default = None, required = True,
                    help = 'Root directory for output files')

parser.add_argument('-l','--logfile', type = str, default = 'logfile.txt', required = False,
                    help = 'Name of log file written to outdir, default is logfile.txt')

parser.add_argument('-c','--clobber', default=False, required=False, action='store_true',
                    help='Clobber (overwrite) BAL catalog if it already exists?')

parser.add_argument('-v','--verbose', default=False, required=False, action='store_true',
                    help = 'Provide verbose output?')

args  = parser.parse_args()

if debug: 
    args.verbose=True
    

# Root directory for input data: 
dataroot = os.path.join(os.getenv("DESI_SPECTRO_REDUX"), args.release, "healpix", args.survey, args.moon) 
    
# Root directory for output catalogs: 
outroot = os.path.join(args.outdir, args.release, "healpix", args.survey, args.moon)
pmmkdir(outroot)

# All possible healpix --
healpixdirs = glob(dataroot + "/*/*")
healpixels = []  # list of all available healpix
for healpixdir in healpixdirs:
    healpixels.append(healpixdir[healpixdir.rfind('/')+1::])

# Requested healpix
inputhealpixels = args.healpix

zfileroot = args.zfileroot

# Check that all requested healpix exist
if inputhealpixels is not None:
    for inputhealpixel in inputhealpixels: 
        assert(inputhealpixel in healpixels), "Healpix {} not available".format(inputhealpixel)
else:
    inputhealpixels = healpixels

# Create/confirm output healpix directories exist
for inputhealpixel in inputhealpixels: 
    healpixdir = os.path.join(outroot, inputhealpixel[:len(inputhealpixel)-2], inputhealpixel) 
    pmmkdir(healpixdir) 

# List of healpix that caused issues for by hand rerun.
issuehealpixels = []
errortypes = []

f = open(outroot + "/" + args.logfile, 'a') 
lastupdate = "Last updated {0} UT by {1}\n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), os.getlogin())
commandline = " ".join(sys.argv)
f.write(lastupdate)
f.write(commandline+'\n')

# For each healpix in inputhealpixels, identify the coadd data
# and run desibalfilder
# for healpix in inputhealpixels: 
for healpix in ['11195']: 
    coaddfilename = "coadd-{0}-{1}-{2}.fits".format(args.survey, args.moon, healpix) 
    balfilename = coaddfilename.replace('coadd-', 'baltable-')

    indir = os.path.join(dataroot, healpix[:len(healpix)-2], healpix)
    outdir = os.path.join(outroot, healpix[:len(healpix)-2], healpix)

    coaddfile = os.path.join(indir, coaddfilename) 
    balfile = os.path.join(outdir, balfilename) 

    if args.altzdir is not None: 
        if zfileroot is None:
            zfileroot = 'redrock'
        altzfilename = "{0}-{1}-{2}-{3}.fits".format(zfileroot, args.survey, args.moon, healpix) 
        altzdir = os.path.join(args.altzdir, healpix[:len(healpix)-2], healpix) 
        altzfile = os.path.join(altzdir, altzfilename) 

    if args.verbose:
        print("Coadd file: ", coaddfile)
        print("BAL file: ", balfile)
        if args.altzdir is not None: 
            print("Redshift file: ", altzfile)

    if not os.path.isfile(balfile) or args.clobber:
        try:
            db.desibalfinder(coaddfile, altbaldir=outdir, altzdir=args.altzdir, zfileroot=zfileroot, overwrite=args.clobber, verbose=args.verbose, release=args.release)
        except:
            print("An error occured at healpix {}. Adding healpix to issuehealpixels list.".format(healpix))
            errortype = sys.exc_info()[0]
            issuehealpixels.append(healpix)
            errortypes.append(errortype)

outstr = "List of healpix with errors and error types: \n"
f.write(outstr) 
for i in range(len(issuehealpixels)):
    f.write("{} : {}\n".format(issuehealpixels[i], errortypes[i]))

f.close()
