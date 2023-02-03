#!/usr/bin/env python
"""

baltools.splitafterburner_hp
=========================

Split an afterburner redshift catalog into separate, healpix-based redshift catalogs in order 
to run the balfilder on individual healpix files

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
                                     description="""Split an afterburner into redshift catalogs organized by healpix""")

parser.add_argument('-q', '--qsocat', type = str, default = None, required = True,
                    help = 'Input QSO catalog')

parser.add_argument('-a','--altzdir', type=str, default = None, required=True,
                    help='Path to directory structure with healpix-based afterburner redshift catalogs')

parser.add_argument('-z','--zfileroot', type=str, default="zafter", 
                    required=False, help='Root name of healpix-based afterburner redshift catalogs')

parser.add_argument('-s', '--survey', type = str, default = 'main', required = False,
                    help = 'Survey subdirectory [sv1, sv2, sv3, main], default is main')

parser.add_argument('-m', '--moon', type = str, default = 'dark', required = False,
                    help = 'Moon brightness [bright, dark], default is dark')

parser.add_argument('-l','--logfile', type = str, default = 'logfile-{survey}-{moon}.txt', required = False,
                    help = 'Name of log file written to outdir, default is logfile-{survey}-{moon}.txt')

parser.add_argument('-c','--clobber', default=False, required=False, action='store_true', 
                    help='Clobber (overwrite) BAL catalog if it already exists?')

parser.add_argument('-v','--verbose', default=False, required=False, action='store_true', 
                    help = 'Provide verbose output?')


args  = parser.parse_args()

# Check the QSO catalog exists
if not os.path.isfile(args.qsocat):
    print("Error: cannot find ", args.qsocat)
    exit(1)
    
    
# # Full path to the output QSO+BAL catalog
# outcat = os.path.join(args.baldir, args.outcatfile) 
# 
# # Add empty BAL cols to qso cat and writes to outcat.
# # Stores return value (BAL card names) in cols
# cols = pt.inittab(args.qsocat, outcat)
# # # Want to manually set this to -1 to show that it is not populated
# # cols.remove('BAL_PROB')
 
qhdu = fits.open(args.qsocat)
qcat = qhdu[1].data

# Calculate healpix for every QSO 
healpixels = hp.ang2pix(64, qcat['TARGET_RA'], qcat['TARGET_DEC'], lonlat=True, nest=True)

# Construct a list of unique healpix pixels
healpixlist = np.unique(healpixels)

# Construct an array of indices for the QSO catalog
hindxs = np.arange(0, len(qcat), dtype=int)

if args.verbose: 
    print("Found {0} entries with {1} unique healpix".format(len(healpixels), len(healpixlist)))

# logfile = os.path.join(args.baldir, args.logfile) 
logfile = os.path.join(args.altzdir, "logfile-{0}-{1}.txt".format(args.survey, args.moon))
f = open(logfile, 'a')
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

# lastupdate = "Last updated {0} UT by {1}\n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), os.getlogin())

commandline = " ".join(sys.argv)
f.write(lastupdate)
f.write(commandline+'\n')
outstr = "Healpix NQSOs Nmatches \n"
f.write(outstr)

for healpix in healpixlist: 
    zfilename = "{0}-{1}-{2}-{3}.fits".format(args.zfileroot, args.survey, args.moon, healpix) 
    zdir = os.path.join(args.altzdir, "healpix", args.survey, args.moon, str(healpix)[:len(str(healpix))-2], str(healpix))
    pmmkdir(zdir) 
    zfile = os.path.join(args.altzdir, "healpix", args.survey, args.moon, str(healpix)[:len(str(healpix))-2], str(healpix), zfilename)
    # bhdu = fits.open(balfile) 
    # bcat = bhdu['BALCAT'].data
     
    # Check if the file already exists
    if os.path.isfile(zfile) and args.clobber == False:
        print("Error: {0} exists and clobber = False".format(zfile))
        continue

    hmask = healpixels == healpix  # mask of everything in qcat in this healpix

    nqsos = np.sum(hmask)
    spectypes = np.empty(len(hmask), dtype='<U3')
    spectypes.fill('QSO')

    if args.verbose: 
        print("Processing healpix {0} with {1} into file {2} ".format(healpix, nqsos, zfile))

    indxs = hindxs[hmask] # indices in qcat that are in pixel healpix

    # Create the fits table:
    col0 = fits.Column(name='TARGETID', format='K', array=qcat['TARGETID'][hmask])
    col1 = fits.Column(name='TARGET_RA', format='E', array=qcat['TARGET_RA'][hmask])
    col2 = fits.Column(name='TARGET_DEC', format='E', array=qcat['TARGET_DEC'][hmask])
    col3 = fits.Column(name='Z', format='E', array=qcat['Z'][hmask])
    col4 = fits.Column(name='ZERR', format='E', array=qcat['ZERR'][hmask])
    col5 = fits.Column(name='ZWARN', format='E', array=qcat['ZWARN'][hmask])
    col6 = fits.Column(name='SPECTYPE', format='6A', array=spectypes[hmask])

    # print(np.where(col6['SPECTYPE'] != 'QSO') )

    ztabhdu = fits.BinTableHDU.from_columns([col0, col1, col2, col3, col4, col5, col6])
    # ztabhdu = fits.BinTableHDU.from_columns([col0, col1, col2, col3])
    ztabhdu.header['EXTNAME'] = 'REDSHIFTS'

    ztabhdu.writeto(zfile, overwrite=args.clobber)  
     
    if args.verbose:
        print("Wrote output file {0}".format(zfile))

    f.write("{0}: {1}\n".format(zfilename, nqsos))

f.close()

print("Finished")
