#!/usr/bin/env python3
"""
Update existing QSO catalogue with BAL information from individual healpix catalogs.

This module utilizes functions from popqsotab.py to add empty BAL columns to existing
QSO catalogues and populate them with information from individual BAL catalogs generated
by runbalfinder.py. It processes healpix-based BAL catalogs and matches them to QSOs
in the main catalog using TARGETID matching.

The module is optimized for NERSC systems with high core counts and includes
parallel processing capabilities for efficient handling of large datasets.

Author: Simon Filbert (2021), Optimized version for NERSC systems
License: DESI Collaboration License
"""

import os
import sys
import numpy as np
from astropy.io import fits
import healpy as hp
import fitsio
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import argparse
from time import gmtime, strftime

from baltools import balconfig as bc
from baltools import popqsotab as pt
from baltools import utils

# Set DESI environment
os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux'

# Define BAL column names
BAL_COLS = [
    'PCA_COEFFS', 'PCA_CHI2', 'BAL_PROB', 'BI_CIV', 'ERR_BI_CIV', 'NCIV_2000',
    'VMIN_CIV_2000', 'VMAX_CIV_2000', 'POSMIN_CIV_2000', 'FMIN_CIV_2000',
    'AI_CIV', 'ERR_AI_CIV', 'NCIV_450', 'VMIN_CIV_450', 'VMAX_CIV_450',
    'POSMIN_CIV_450', 'FMIN_CIV_450', 'BI_SIIV', 'ERR_BI_SIIV', 'NSIIV_2000',
    'VMIN_SIIV_2000', 'VMAX_SIIV_2000', 'POSMIN_SIIV_2000', 'FMIN_SIIV_2000',
    'AI_SIIV', 'ERR_AI_SIIV', 'NSIIV_450', 'VMIN_SIIV_450', 'VMAX_SIIV_450',
    'POSMIN_SIIV_450', 'FMIN_SIIV_450', 'SNR_CIV', 'SNR_REDSIDE', 'SNR_FOREST'
]

def balcopy(qinfo: np.ndarray, binfo: np.ndarray) -> None:
    """
    Copy BAL properties from BAL catalog entry to QSO catalog entry.
    
    Parameters
    ----------
    qinfo : np.ndarray
        QSO catalog entry to be updated
    binfo : np.ndarray
        BAL catalog entry containing the properties to copy
    """

    # Get the available column names from the source BAL catalog entry
    source_cols = binfo.dtype.names

    for balcol in BAL_COLS: 
        # Only copy the column if it exists in the source file
        if balcol in source_cols:
            qinfo[balcol] = binfo[balcol]
    
    qinfo['BALMASK'] = 0


def calculate_healpix_vectorized(ra: np.ndarray, dec: np.ndarray, nside: int = 64) -> np.ndarray:
    """
    Calculate healpix indices for all coordinates using vectorized operations.
    
    Parameters
    ----------
    ra : np.ndarray
        Right ascension coordinates in degrees
    dec : np.ndarray
        Declination coordinates in degrees
    nside : int, optional
        HEALPix nside parameter, by default 64
        
    Returns
    -------
    np.ndarray
        Array of healpix indices for each coordinate pair
    """
    return hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)


def match_targets_vectorized(qso_targetids: np.ndarray, bal_targetids: np.ndarray) -> List[Tuple[int, int]]:
    """
    Match targets using vectorized operations for better performance.
    
    Parameters
    ----------
    qso_targetids : np.ndarray
        Array of TARGETIDs from QSO catalog
    bal_targetids : np.ndarray
        Array of TARGETIDs from BAL catalog
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (qso_index, bal_index) tuples for matched targets
    """
    # Create a mapping from targetid to index
    qso_map = {tid: idx for idx, tid in enumerate(qso_targetids)}
    
    # Find matches using vectorized operations
    matches = []
    for bal_idx, bal_tid in enumerate(bal_targetids):
        if bal_tid in qso_map:
            matches.append((qso_map[bal_tid], bal_idx))
    
    return matches


def process_healpix_batch_with_qcat(healpix_batch: List[int], args: argparse.Namespace, 
                                   qcat: np.ndarray, healpixels: np.ndarray, 
                                   baldir: str, survey: str, moon: str, 
                                   mock: bool) -> List[Tuple[int, np.ndarray]]:
    """
    Process a batch of healpix pixels with access to the full QSO catalog.
    
    Parameters
    ----------
    healpix_batch : List[int]
        List of healpix indices to process
    args : argparse.Namespace
        Command line arguments
    qcat : np.ndarray
        QSO catalog data
    healpixels : np.ndarray
        Array of healpix indices for all QSOs
    baldir : str
        Directory containing BAL catalogs
    survey : str
        Survey name
    moon : str
        Moon brightness setting
    mock : bool
        Whether processing mock data
        
    Returns
    -------
    List[Tuple[int, np.ndarray]]
        List of (qcat_index, bal_data) tuples for modifications to apply
    """
    modifications = []  # List of (qcat_index, bal_data) tuples
    
    for healpix in healpix_batch:
        hpdir = utils.gethpdir(str(healpix))
        
        if mock: 
            balfilename = f"baltable-16-{healpix}.fits"
            balfile = Path(baldir) / 'spectra-16' / hpdir / str(healpix) / balfilename
        else: 
            balfilename = f"baltable-{survey}-{moon}-{healpix}.fits"
            balfile = Path(baldir) / "healpix" / survey / moon / hpdir / str(healpix) / balfilename
        
        try: 
            # Use fitsio for faster reading
            bcat = fitsio.read(str(balfile), ext='BALCAT')
        except (FileNotFoundError, OSError):
            if args.verbose:
                print(f"Warning: Did not find {balfile}")
            continue

        # Find QSOs in this healpix
        hmask = healpixels == healpix
        healpix_qso_indices = np.arange(len(healpixels))[hmask]
        
        if len(healpix_qso_indices) == 0:
            continue

        # Get the actual TARGETIDs for QSOs in this healpix
        healpix_targetids = qcat['TARGETID'][healpix_qso_indices]

        if args.verbose:
            print(f"Processing healpix {healpix}: {len(healpix_qso_indices)} QSOs, {len(bcat)} BAL entries")

        # Use vectorized matching for better performance
        matches = match_targets_vectorized(healpix_targetids, bcat['TARGETID'])
        
        if args.verbose and len(matches) > 0:
            print(f"  Found {len(matches)} matches in healpix {healpix}")
        
        # Store modifications to apply later
        for qidx, bidx in matches:
            qcat_index = healpix_qso_indices[qidx]
            modifications.append((qcat_index, bcat[bidx]))
    
    return modifications


def setup_logging(args: argparse.Namespace, baldir: str) -> Path:
    """
    Setup logging file and write header.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    baldir : str
        BAL directory path
        
    Returns
    -------
    Path
        Path to the log file
    """
    if args.mock: 
        logfile = Path(baldir) / "logfile-mock.txt"
    else: 
        logfile = Path(baldir) / f"logfile-{args.survey}-{args.moon}.txt"

    # Write log header
    try:
        lastupdate = f"Last updated {strftime('%Y-%m-%d %H:%M:%S', gmtime())} UT by {os.getlogin()}\n"
    except OSError:
        try:
            lastupdate = f"Last updated {strftime('%Y-%m-%d %H:%M:%S', gmtime())} UT by {os.getenv('USER')}\n"
        except (TypeError, KeyError):
            try:
                lastupdate = f"Last updated {strftime('%Y-%m-%d %H:%M:%S', gmtime())} UT by {os.getenv('LOGNAME')}\n"
            except (TypeError, KeyError):
                lastupdate = f"Last updated {strftime('%Y-%m-%d %H:%M:%S', gmtime())} UT\n"
    
    with open(logfile, 'a') as f:
        f.write(lastupdate)
        f.write(" ".join(sys.argv) + '\n')
        f.write("Healpix NQSOs Nmatches\n")
    
    return logfile


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Update existing QSO catalogue with BAL information"
    )

    parser.add_argument('-q', '--qsocat', type=str, default=None, required=True,
                        help='Input QSO catalog')

    parser.add_argument('-b', '--baldir', type=str, default=None, required=True,
                        help='Path to directory structure with individual BAL catalogs')

    parser.add_argument('-o', '--outcatfile', type=str, default="qso-balcat.fits", 
                        required=False, help='Output QSO+BAL catalog')

    parser.add_argument('-s', '--survey', type=str, default='main', required=False,
                        help='Survey subdirectory [sv1, sv2, sv3, main], default is main')

    parser.add_argument('-m', '--moon', type=str, default='dark', required=False,
                        help='Moon brightness [bright, dark], default is dark')

    parser.add_argument('--mock', default=False, required=False, action='store_true',
                        help='Mock catalog?, default is False')
                
    parser.add_argument('-l', '--logfile', type=str, default='logfile-{survey}-{moon}.txt', required=False,
                        help='Name of log file written to outdir, default is logfile-{survey}-{moon}.txt')

    parser.add_argument('-c', '--clobber', default=False, required=False, action='store_true', 
                        help='Clobber (overwrite) BAL catalog if it already exists?')

    parser.add_argument('-v', '--verbose', default=False, required=False, action='store_true', 
                        help='Provide verbose output?')

    parser.add_argument('-t', '--alttemp', default=False, required=False, action='store_true',
                        help='Use alternate components made by Allyson Brodzeller')

    parser.add_argument('--nproc', type=int, default=256, required=False,
                        help='Number of processes for parallel processing (default: 256 for NERSC nodes)')

    parser.add_argument('--chunk-size', type=int, default=100, required=False,
                        help='Chunk size for parallel processing (default: 100)')

    return parser.parse_args()


def main() -> None:
    """
    Main function to update QSO catalog with BAL information.
    
    This function:
    1. Reads the input QSO catalog and initializes BAL columns
    2. Calculates healpix indices for all QSOs
    3. Processes healpix-based BAL catalogs in parallel batches
    4. Matches QSOs to BAL entries using TARGETID
    5. Copies BAL properties to the QSO catalog
    6. Applies redshift range masking
    7. Writes the final output catalog
    """
    args = parse_arguments()
    
    # Check the QSO catalog exists
    qsocat_path = Path(args.qsocat)
    if not qsocat_path.exists():
        print(f"Error: cannot find {qsocat_path}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Reading QSO catalog: {qsocat_path}")
    
    # Full path to the output QSO+BAL catalog
    outcat = Path(args.outcatfile)

    # Add empty BAL cols to qso cat and writes to outcat.
    # Stores return value (BAL card names) in cols
    try:
        cols = pt.inittab(str(qsocat_path), str(outcat), alttemp=args.alttemp)
    except Exception as e:
        print(f"Error initializing BAL columns: {e}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Initialized BAL columns in output catalog: {outcat}")

    # Read QSO catalog using fitsio for better performance
    try:
        qcat = fitsio.read(str(outcat), ext=1)
    except Exception as e:
        print(f"Error reading QSO catalog {outcat}: {e}")
        sys.exit(1)

    if args.verbose:
        print(f"Loaded {len(qcat)} QSOs from catalog")

    # Calculate healpix for every QSO using vectorized operations
    if args.mock: 
        healpixels = calculate_healpix_vectorized(qcat['RA'], qcat['DEC'], nside=16)
    else: 
        healpixels = calculate_healpix_vectorized(qcat['TARGET_RA'], qcat['TARGET_DEC'], nside=64)

    # Construct a list of unique healpix pixels
    healpixlist = np.unique(healpixels)

    if args.verbose: 
        print(f"Found {len(healpixlist)} unique healpix")

    # Setup logging
    logfile = setup_logging(args, args.baldir)

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
                executor.submit(process_healpix_batch_with_qcat, batch, args, qcat, healpixels, 
                               args.baldir, args.survey, args.moon, args.mock): batch 
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
        all_modifications = process_healpix_batch_with_qcat(
            healpixlist, args, qcat, healpixels, args.baldir, args.survey, args.moon, args.mock
        )

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
    zmask = zmask * (qcat['Z'] <= bc.BAL_ZMAX)
    zmask = ~zmask  # check to True for out of redshift range
    zbit = 2 * np.ones(len(zmask), dtype=np.ubyte)  # bitmask for out of redshift range
    qcat['BALMASK'][zmask] += zbit[zmask]

    # Write final catalog using fitsio for better performance
    try:
        fitsio.write(str(outcat), qcat, extname='ZCATALOG', clobber=True)
    except Exception as e:
        print(f"Error writing final catalog {outcat}: {e}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Wrote final catalog: {outcat}")
        print(f"Total matches: {total_matches}")

    print(f"Wrote {outcat}")


if __name__ == "__main__":
    main() 
