#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

import argparse
from baltools import desibal as db


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Run balfinder on DESI data""")

parser.add_argument('-i','--in-file', type = str, nargs='*', required = False,
                    help = 'Path to input files to process')

parser.add_argument('--in-dir', type = str, required = False,
                    help = 'Path to directory containing all files to process')

#allow to specify the output directory
#parser.add_argument('-o','--out-dir', type = str, default = None, required = False,
#                    help = 'Path for output BAL catalog file')

parser.add_argument('-r','--redo', type = bool, default = False, required = False,
                    help = 'Redo (overwrite) BAL catalog if it already exists?')

parser.add_argument('-v','--verbose', type = bool, default = False, required = False,
                    help = 'Provide verbose output?')

args  = parser.parse_args()

# Identify the files to process
if args.in_file is not None:
    files= args.in_file
else:
    #Try to read and process all files in spectra directory
    files = glob(args.in_dir+"/*/*/spectra*.fits")

##TODO:Add a few lines here to skip already procesed files

#Run the BAL finder in each file...
for specfilename in files:
     print("Procesisng:",specfilename)
     db.desibalfinder(specfilename, overwrite=args.redo, verbose=False)


