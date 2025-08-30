#!/usr/bin/env python
"""

baltools.appendbalinfo_hp
=========================

Utilizes functinos from popqsotab.py to add empty BAL columns to existing
QSO catalogue and add information from baltables to new catalogue.
runbalfinder.py tables

2021 Original code by Simon Filbert
OPTIMIZED VERSION for NERSC systems with high core counts

"""

import os
import sys
import numpy as np
from astropy.io import fits
import healpy as hp
import fitsio
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import argparse
from time import gmtime, strftime

from baltools import balconfig as bc
# from baltools import fitbal
from baltools import popqsotab as pt
from baltools import utils


balcols = ['PCA_COEFFS', 'PCA_CHI2', 'BAL_PROB', 'BI_CIV', 'ERR_BI_CIV', 'NCIV_2000', 'VMIN_CIV_2000', 'VMAX_CIV_2000', 'POSMIN_CIV_2000', 'FMIN_CIV_2000', 'AI_CIV', 'ERR_AI_CIV', 'NCIV_450', 'VMIN_CIV_450', 'VMAX_CIV_450', 'POSMIN_CIV_450', 'FMIN_CIV_450', 'BI_SIIV', 'ERR_BI_SIIV', 'NSIIV_2000', 'VMIN_SIIV_2000', 'VMAX_SIIV_2000', 'POSMIN_SIIV_2000', 'FMIN_SIIV_2000', 'AI_SIIV', 'ERR_AI_SIIV', 'NSIIV_450', 'VMIN_SIIV_450', 'VMAX_SIIV_450', 'POSMIN_SIIV_450', 'FMIN_SIIV_450', 'SNR_CIV']

def balcopy(qinfo, binfo):
    for balcol in balcols: 
        qinfo[balcol] = binfo[balcol]
    qinfo['BALMASK'] = 0

def calculate_healpix_vectorized(ra, dec, nside=64):
    """Calculate healpix for all coordinates at once using vectorized operations"""
    return hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)

def match_targets_vectorized(qso_targetids, bal_targetids):
    """Match targets using vectorized operations for better performance"""
    # Create a mapping from targetid to index
    qso_map = {tid: idx for idx, tid in enumerate(qso_targetids)}
    
    # Find matches using vectorized operations
    matches = []
    for bal_idx, bal_tid in enumerate(bal_targetids):
        if bal_tid in qso_map:
            matches.append((qso_map[bal_tid], bal_idx))
    
    return matches

def process_healpix_batch(healpix_batch, args, qcat_indices, healpixels, baldir, survey, moon, mock):
    """Process a batch of healpix pixels and return modifications to apply"""
    modifications = []  # List of (qcat_index, bal_data) tuples
    
    for healpix in healpix_batch:
        hpdir = utils.gethpdir(str(healpix))
        
        if mock: 
            balfilename = f"baltable-16-{healpix}.fits"
            balfile = os.path.join(baldir, 'spectra-16', hpdir, str(healpix), balfilename)
        else: 
            balfilename = f"baltable-{survey}-{moon}-{healpix}.fits"
            balfile = os.path.join(baldir, "healpix", survey, moon, hpdir, str(healpix), balfilename)
        
        try: 
            # Use fitsio for faster reading
            bcat = fitsio.read(balfile, ext='BALCAT')
        except (FileNotFoundError, OSError):
            continue

        # Find QSOs in this healpix
        hmask = healpixels == healpix
        healpix_qso_indices = qcat_indices[hmask]
        healpix_targetids = np.arange(len(healpixels))[hmask]  # Use indices for matching
        
        if len(healpix_qso_indices) == 0:
            continue

        # Use vectorized matching for better performance
        matches = match_targets_vectorized(healpix_targetids, bcat['TARGETID'])
        
        # Store modifications to apply later
        for qidx, bidx in matches:
            qcat_index = healpix_qso_indices[qidx]
            modifications.append((qcat_index, bcat[bidx]))
    
    return modifications

os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Update existing QSO catalogue with BAL information""")

parser.add_argument('-q', '--qsocat', type = str, default = None, required = True,
                    help = 'Input QSO catalog')

parser.add_argument('-b','--baldir', type=str, default = None, required=True,
                    help='Path to directory structure with individual BAL catalogs')

parser.add_argument('-o','--outcatfile', type=str, default="qso-balcat.fits", 
                    required=False, help='Output QSO+BAL catalog')

parser.add_argument('-s', '--survey', type = str, default = 'main', required = False,
                    help = 'Survey subdirectory [sv1, sv2, sv3, main], default is main')

parser.add_argument('-m', '--moon', type = str, default = 'dark', required = False,
                    help = 'Moon brightness [bright, dark], default is dark')

parser.add_argument('--mock', default=False, required = False, action='store_true',
                    help = 'Mock catalog?, default is False')
                
parser.add_argument('-l','--logfile', type = str, default = 'logfile-{survey}-{moon}.txt', required = False,
                    help = 'Name of log file written to outdir, default is logfile-{survey}-{moon}.txt')

parser.add_argument('-c','--clobber', default=False, required=False, action='store_true', 
                    help='Clobber (overwrite) BAL catalog if it already exists?')

parser.add_argument('-v','--verbose', default=False, required=False, action='store_true', 
                    help = 'Provide verbose output?')

parser.add_argument('-t','--alttemp', default=False, required=False, action='store_true',
                    help = 'Use alternate components made by Allyson Brodzeller')

parser.add_argument('--nproc', type=int, default=64, required=False,
                    help='Number of processes for parallel processing (default: 64)')

parser.add_argument('--chunk-size', type=int, default=100, required=False,
                    help='Chunk size for parallel processing (default: 100)')

args  = parser.parse_args()

def main():
    # Check the QSO catalog exists
    if not os.path.isfile(args.qsocat):
        print("Error: cannot find ", args.qsocat)
        exit(1)
    
    if args.verbose:
        print(f"Reading QSO catalog: {args.qsocat}")
    
    # Full path to the output QSO+BAL catalog
    outcat = os.path.join(args.outcatfile) 

    # Add empty BAL cols to qso cat and writes to outcat.
    # Stores return value (BAL card names) in cols
    cols = pt.inittab(args.qsocat, outcat, alttemp=args.alttemp)
    
    if args.verbose:
        print(f"Initialized BAL columns in output catalog: {outcat}")

    # Read QSO catalog using fitsio for better performance
    qcat = fitsio.read(outcat, ext=1)

    if args.verbose:
        print(f"Loaded {len(qcat)} QSOs from catalog")

    # Calculate healpix for every QSO using vectorized operations
    if args.mock: 
        healpixels = calculate_healpix_vectorized(qcat['RA'], qcat['DEC'], nside=16)
    else: 
        healpixels = calculate_healpix_vectorized(qcat['TARGET_RA'], qcat['TARGET_DEC'], nside=64)

    # Construct a list of unique healpix pixels
    healpixlist = np.unique(healpixels)

    # Construct an array of indices for the QSO catalog
    qcat_indices = np.arange(0, len(qcat), dtype=int)

    if args.verbose: 
        print(f"Found {len(healpixlist)} unique healpix")

    # Setup logging
    if args.mock: 
        logfile = os.path.join(args.baldir, "logfile-mock.txt")
    else: 
        logfile = os.path.join(args.baldir, f"logfile-{args.survey}-{args.moon}.txt")

    # Write log header
    try:
        lastupdate = f"Last updated {strftime('%Y-%m-%d %H:%M:%S', gmtime())} UT by {os.getlogin()}\n"
    except:
        try:
            lastupdate = f"Last updated {strftime('%Y-%m-%d %H:%M:%S', gmtime())} UT by {os.getenv('USER')}\n"
        except:
            try:
                lastupdate = f"Last updated {strftime('%Y-%m-%d %H:%M:%S', gmtime())} UT by {os.getenv('LOGNAME')}\n"
            except:
                lastupdate = f"Last updated {strftime('%Y-%m-%d %H:%M:%S', gmtime())} UT\n"
    
    with open(logfile, 'a') as f:
        f.write(lastupdate)
        f.write(" ".join(sys.argv) + '\n')
        f.write("Healpix NQSOs Nmatches\n")

    # Process healpix in parallel batches
    all_modifications = []
    total_matches = 0
    
    if args.nproc > 1 and len(healpixlist) > 1:
        # Split healpix into batches for parallel processing
        batch_size = max(1, min(args.chunk_size, len(healpixlist) // (args.nproc * 4)))
        healpix_batches = [healpixlist[i:i+batch_size] for i in range(0, len(healpixlist), batch_size)]
        
        if args.verbose:
            print(f"Processing {len(healpixlist)} healpix in {len(healpix_batches)} batches using {args.nproc} processes")
        
        with ProcessPoolExecutor(max_workers=args.nproc) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(process_healpix_batch, batch, args, qcat_indices, healpixels, args.baldir, args.survey, args.moon, args.mock): batch 
                for batch in healpix_batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_modifications = future.result()
                    all_modifications.extend(batch_modifications)
                    
                    if args.verbose:
                        batch = future_to_batch[future]
                        print(f"Completed batch with {len(batch)} healpix, found {len(batch_modifications)} matches")
                except Exception as e:
                    batch = future_to_batch[future]
                    print(f"Error processing batch with {len(batch)} healpix: {e}")
    else:
        # Sequential processing
        all_modifications = process_healpix_batch(healpixlist, args, qcat_indices, healpixels, args.baldir, args.survey, args.moon, args.mock)

    # Apply all modifications to the QSO catalog
    if args.verbose:
        print(f"Applying {len(all_modifications)} BAL property modifications to QSO catalog")
    
    for qcat_index, bal_data in all_modifications:
        balcopy(qcat[qcat_index], bal_data)
        total_matches += 1

    # Write results to log file
    with open(logfile, 'a') as f:
        # Group modifications by healpix for logging
        healpix_matches = defaultdict(int)
        for qcat_index, _ in all_modifications:
            healpix = healpixels[qcat_index]
            healpix_matches[healpix] += 1
        
        for healpix, nmatch in healpix_matches.items():
            if args.mock:
                balfilename = f"baltable-16-{healpix}.fits"
            else:
                balfilename = f"baltable-{args.survey}-{args.moon}-{healpix}.fits"
            f.write(f"{balfilename}: {nmatch} {nmatch}\n")

    # Apply redshift range mask
    zmask = qcat['Z'] >= bc.BAL_ZMIN
    zmask = zmask*(qcat['Z'] <= bc.BAL_ZMAX)
    zmask = ~zmask # check to True for out of redshift range
    zbit = 2*np.ones(len(zmask), dtype=np.ubyte) # bitmask for out of redshift range
    qcat['BALMASK'][zmask] += zbit[zmask]

    # Write final catalog using fitsio for better performance
    fitsio.write(outcat, qcat, extname='ZCATALOG', clobber=True)
    
    if args.verbose:
        print(f"Wrote final catalog: {outcat}")
        print(f"Total matches: {total_matches}")

    print(f"Wrote {outcat}")

if __name__ == "__main__":
    main() 
