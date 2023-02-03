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
from baltools import utils

from multiprocessing import Pool

os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux'

def parse(options=None): 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Run balfinder on DESI data""")

    parser.add_argument('-hp','--healpix', nargs='*', default = None, required=False,
                    help='List of healpix number(s) to process - default is all')

    parser.add_argument('-r', '--release', type = str, default = 'everest', required = False,
                    help = 'Data release subdirectory, default is everest')

    parser.add_argument('-a','--altzdir', type=str, default = None, required=True,
                    help='Path to directory structure with healpix-based afterburner redshift catalogs')

    parser.add_argument('-z','--zfileroot', type=str, default='zafter', required=False, 
                    help='Root name of healpix-based afterburner redshift catalogs')

    parser.add_argument('-s', '--survey', type = str, default = 'main', required = False,
                    help = 'Survey subdirectory [sv1, sv2, sv3, main], default is main')

    parser.add_argument('-m', '--moon', type = str, default = 'dark', required = False,
                    help = 'Moon brightness [bright, dark], default is dark')

    parser.add_argument('-o','--outdir', type = str, default = None, required = True,
                    help = 'Root directory for output files')

    parser.add_argument('-l','--logfile', type = str, default = 'logfile.txt', required = False,
                    help = 'Name of log file written to outdir, default is logfile.txt')

    parser.add_argument('--nproc', type=int, default=64, required=False,
                        help='Number of processes')

    parser.add_argument('-c','--clobber', default=False, required=False, action='store_true',
                    help='Clobber (overwrite) BAL catalog if it already exists?')

    parser.add_argument('-v','--verbose', default=False, required=False, action='store_true',
                    help = 'Provide verbose output?')

    if options is None: 
        args  = parser.parse_args()
    else: 
        args  = parser.parse_args(options)

    return args 


def main(args=None): 

    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    # Root directory for input data: 
    dataroot = os.path.join(os.getenv("DESI_SPECTRO_REDUX"), args.release, "healpix", args.survey, args.moon) 
        
    # Root directory for output catalogs: 
    outroot = os.path.join(args.outdir, "healpix", args.survey, args.moon)
    utils.pmmkdir(outroot)
    
    # All possible healpix --
    healpixdirs = glob(dataroot + "/*/*")
    healpixels = []  # list of all available healpix
    for healpixdir in healpixdirs:
        healpixels.append(healpixdir[healpixdir.rfind('/')+1::])
    
    # Requested healpix
    inputhealpixels = args.healpix
    
    zfileroot = args.zfileroot
    altzdir = None
    
    # Check that all requested healpix exist
    if inputhealpixels is not None:
        for inputhealpixel in inputhealpixels: 
            assert(str(inputhealpixel) in healpixels), "Healpix {} not available".format(inputhealpixel)
    else:
        inputhealpixels = healpixels
    
    # Create/confirm output healpix directories exist
    for inputhealpixel in inputhealpixels: 
        hpdir = utils.gethpdir(inputhealpixel)
        healpixdir = os.path.join(outroot, hpdir, inputhealpixel) 
        utils.pmmkdir(healpixdir) 
    
    # List of healpix that caused issues for by hand rerun.
    issuehealpixels = []
    errortypes = []
    
    outlog = os.path.join(outroot, args.logfile)
    f = open(outlog, 'a') 
    try: 
        lastupdate = "Last updated {0} UT by {1}\n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), os.getlogin())
    except: 
        try: 
            lastupdate = "Last updated {0} UT by {1}\n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), os.getenv('USER'))
        except: 
            try: 
                lastupdate = "Last updated {0} UT by {1}\n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), os.getenv('LOGNAME'))
            except: 
                print("Error with tagging log file") 
                lastupdate = "Last updated {0} UT \n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())) 
    
    commandline = " ".join(sys.argv)
    f.write(lastupdate)
    f.write(commandline+'\n')

    if args.nproc > 1: 
        func_args = [ {"healpix": healpix , \
                       "args": args, \
                       "healpixels": inputhealpixels, \
                       "dataroot": dataroot, \
                       "outroot": outroot \
                    } for ih,healpix in enumerate(inputhealpixels) ]
        with Pool(args.nproc) as pool: 
            results = pool.map(_func, func_args)
    else: 
        for ih,healpix in enumerate(inputhealpixels) : 
            results = findbals_one_healpix(healpix)

    lresults = list(results) 

    for i in range(len(lresults)): 
        if lresults[i][1] is not None: 
            issuehealpixels.append(lresults[i][0]) 
            errortypes.append(lresults[i][1]) 

    outstr = "List of healpix with errors and error types: \n"
    f.write(outstr) 
    for i in range(len(issuehealpixels)):
        f.write("{} : {}\n".format(issuehealpixels[i], errortypes[i]))
    
    f.close()
    print(f"Wrote output log {outlog}") 


def _func(arg): 
    """ Used for multiprocessing.Pool """
    return findbals_one_healpix(**arg)


def findbals_one_healpix(healpix, args, healpixels, dataroot, outroot): 

    skiphealpix = False
    coaddfilename = "coadd-{0}-{1}-{2}.fits".format(args.survey, args.moon, healpix) 
    balfilename = coaddfilename.replace('coadd-', 'baltable-')

    hpdir = utils.gethpdir(healpix) 
    indir = os.path.join(dataroot, hpdir, healpix)
    outdir = os.path.join(outroot, hpdir, healpix)

    coaddfile = os.path.join(indir, coaddfilename) 
    balfile = os.path.join(outdir, balfilename) 
    zfile = balfile.replace('baltable-', args.zfileroot+"-")

    if args.altzdir is not None: 
        if args.zfileroot is None:
            zfileroot = 'redrock'
        altzdir = os.path.join(args.altzdir, 'healpix', args.survey, args.moon, hpdir, healpix) 

    # Check to see if zfile exists -- if not, skip
    if not os.path.isfile(zfile): 
        skiphealpix = True

    if args.verbose:
        print("Coadd file: ", coaddfile)
        print("BAL file: ", balfile)
        print("zfileroot: ", args.zfileroot)
        if args.altzdir is not None: 
            print("Redshift directory: ", altzdir)
        if skiphealpix: 
            print("Did not find {0}, so skipping healpix {1}".format(zfile, healpix))

    errortype = None
    if not os.path.isfile(balfile) or args.clobber:
        try:
            if not skiphealpix: 
                print(f"About to run db.desibalfinder with verbose={args.verbose} and altbaldir={outdir} and zfileroot={args.zfileroot}") 
                db.desibalfinder(coaddfile, altbaldir=outdir, altzdir=altzdir, zfileroot=args.zfileroot, overwrite=args.clobber, verbose=args.verbose, release=args.release)
            else: 
                errortype = "Did not find redshift catalog {0}".format(zfile)
                # issuehealpixels.append(healpix)
                # errortypes.append(errortype)
        except:
            print("An error occured at healpix {}. Adding healpix to issuehealpixels list.".format(healpix))
            errortype = sys.exc_info()[0]

    return healpix, errortype


if __name__ == "__main__":
    main()

