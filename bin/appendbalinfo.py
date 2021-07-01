import os
import sys
import numpy as np
from astropy.io import fits

import argparse

sys.path.append("/global/homes/s/simonmf/baltools/py")
from baltools import balconfig as bc
from baltools import fitbal
from baltools import initcreate as ic

def pmmkdir(direct): 
    if not os.path.isdir(direct):
        try:
            os.makedirs(direct)
        except PermissionError:
            print("Error: no permission to make directory ", direct)
            exit(1)

os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Update existing QSO catalogue with BAL information""")

parser.add_argument('-q', '--qsocat', type = str, default = ".", required = True,
                    help = 'Input QSO catalogue for which information is to be added to')

parser.add_argument('-b','--baldir', type=str, default=".", required=False,
                    help='Path to directory structure with individual BAL catalogs')

parser.add_argument('-o','--outdir', type = str, default = None, required = True,
                    help = 'Root directory for output catalogue')

parser.add_argument('-f','--filename', type=str, default="qsocat.fits", 
                    required=False, help='Filename of output QSO catalog')

parser.add_argument('-c','--clobber', type=bool, default=False, required=False,
                    help='Clobber (overwrite) BAL catalog if it already exists?')

parser.add_argument('-v','--verbose', type = bool, default = False, required = False,
                    help = 'Provide verbose output?')


args  = parser.parse_args()

outdir   = args.outdir
filename = args.filename
qsocat   = args.qsocat
baldir   = args.baldir

if qsocat == ".":
    print("Input QSO catalogue path is required. Exiting program.")
    exit(1)
if not os.path.isfile(qsocat):
    print("Error: cannot find ", qsocat)
    exit(1)

if baldir == ".":
    print("Root BAL directory path is required. Exiting program.")
    exit(1)
    
# Checks whether outdir exists. If it does not, makes it if permitted.
pmmkdir(outdir)
    
outpath = os.path.join(outdir, filename)
# Adds empty BAL cols to qso cat and writes to outpath.
# Stores return value (BAL card names) in cols
cols = ic.inittab(qsocat, outpath)
# Want to manually set this to -1 to show that it is not populated
cols.remove('BAL_PROB')

cathdu    = fits.open(qsocat)
'''targetids = cathdu[1].data['TARGETID']'''
lencat = len(cathdu[1].data['TARGETID'])

'''for target in targetids:'''
for catindex in range(lencat):
    '''ic.popqsocat(outpath, baldir, target, cols, overwrite=args.clobber, verbose=args.verbose)'''
    ic.popqsocat(outpath, baldir, catindex, cols, overwrite=args.clobber, verbose=args.verbose)
    
    if args.verbose:
        print(("Target ", str(cathdu[1].data['TARGETID'][catindex]), " complete."))
    
    
