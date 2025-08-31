#!/usr/bin/env python3
"""
Split an afterburner redshift catalog into separate, healpix-based redshift catalogs.

This module splits a large QSO catalog into individual healpix-based redshift catalogs
to enable parallel processing of BAL finding on individual healpix files. It's optimized
for NERSC systems with high core counts.

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
import argparse
from time import gmtime, strftime
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from baltools import balconfig as bc
from baltools import popqsotab as pt
from baltools import utils

# Set DESI environment
os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux'


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


def create_healpix_data(qcat: np.ndarray, healpixels: np.ndarray, hmask: np.ndarray, 
                       args: argparse.Namespace) -> np.ndarray:
    """
    Create healpix data structure for a single healpix.
    
    Parameters
    ----------
    qcat : np.ndarray
        QSO catalog data
    healpixels : np.ndarray
        Array of healpix indices for all QSOs
    hmask : np.ndarray
        Boolean mask for QSOs in the current healpix
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    np.ndarray
        Structured array containing QSO data for the healpix
    """
    nqsos = np.sum(hmask)
    spectypes = np.full(nqsos, 'QSO', dtype='<U3')
    
    # Create structured array for FITS output
    dtype = [
        ('TARGETID', 'i8'),
        ('TARGET_RA', 'f8'),
        ('TARGET_DEC', 'f8'),
        ('Z', 'f8'),
        ('ZERR', 'f8'),
        ('ZWARN', 'f8'),
        ('SPECTYPE', 'U6')
    ]
    
    data = np.empty(nqsos, dtype=dtype)
    data['TARGETID'] = qcat['TARGETID'][hmask]
    
    if args.mock:
        data['TARGET_RA'] = qcat['RA'][hmask]
        data['TARGET_DEC'] = qcat['DEC'][hmask]
    else:
        data['TARGET_RA'] = qcat['TARGET_RA'][hmask]
        data['TARGET_DEC'] = qcat['TARGET_DEC'][hmask]
    
    data['Z'] = qcat['Z'][hmask]
    data['ZERR'] = qcat['ZERR'][hmask]
    data['ZWARN'] = qcat['ZWARN'][hmask]
    data['SPECTYPE'] = spectypes
    
    return data


def write_healpix_file(healpix: int, data: np.ndarray, args: argparse.Namespace) -> Tuple[int, int]:
    """
    Write a single healpix file using fitsio for better performance.
    
    Parameters
    ----------
    healpix : int
        Healpix index
    data : np.ndarray
        QSO data for this healpix
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    Tuple[int, int]
        Tuple of (healpix, number_of_qsos)
    """
    hpdir = utils.gethpdir(str(healpix))
    
    if args.mock:
        zfilename = f"{args.zfileroot}-16-{healpix}.fits"
        zdir = Path(args.altzdir) / "spectra-16" / hpdir / str(healpix)
    else:
        zfilename = f"{args.zfileroot}-{args.survey}-{args.moon}-{healpix}.fits"
        zdir = Path(args.altzdir) / "healpix" / args.survey / args.moon / hpdir / str(healpix)
    
    zdir.mkdir(parents=True, exist_ok=True)
    zfile = zdir / zfilename
    
    # Check if file exists and clobber setting
    if zfile.exists() and not args.clobber:
        if args.verbose:
            print(f"Warning: {zfile} exists and clobber = False, skipping")
        return healpix, 0
    
    # Write using fitsio for better performance
    fitsio.write(str(zfile), data, extname='REDSHIFTS', clobber=args.clobber)
    
    if args.verbose:
        print(f"Wrote output file {zfile} with {len(data)} QSOs")
    
    return healpix, len(data)


def process_healpix_batch(healpix_batch: List[int], qcat: np.ndarray, 
                         healpixels: np.ndarray, args: argparse.Namespace) -> List[Tuple[int, int]]:
    """
    Process a batch of healpix pixels.
    
    Parameters
    ----------
    healpix_batch : List[int]
        List of healpix indices to process
    qcat : np.ndarray
        QSO catalog data
    healpixels : np.ndarray
        Array of healpix indices for all QSOs
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (healpix, nqsos) tuples
    """
    results = []
    for healpix in healpix_batch:
        hmask = healpixels == healpix
        if np.sum(hmask) > 0:
            data = create_healpix_data(qcat, healpixels, hmask, args)
            healpix_result, nqsos = write_healpix_file(healpix, data, args)
            results.append((healpix_result, nqsos))
    return results


def setup_logging(args: argparse.Namespace) -> Path:
    """
    Setup logging file and write header.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    Path
        Path to the log file
    """
    if args.mock:
        logfile = Path(args.altzdir) / "logfile-mock.txt"
    else:
        logfile = Path(args.altzdir) / f"logfile-{args.survey}-{args.moon}.txt"
    
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
        description="Split an afterburner into redshift catalogs organized by healpix"
    )

    parser.add_argument('-q', '--qsocat', type=str, default=None, required=True,
                        help='Input QSO catalog')

    parser.add_argument('-a', '--altzdir', type=str, default=None, required=True,
                        help='Path to directory structure with healpix-based afterburner redshift catalogs')

    parser.add_argument('-z', '--zfileroot', type=str, default="zafter", 
                        required=False, help='Root name of healpix-based afterburner redshift catalogs')

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

    parser.add_argument('--nproc', type=int, default=64, required=False,
                        help='Number of processes for parallel processing (default: 64)')

    parser.add_argument('--chunk-size', type=int, default=1000, required=False,
                        help='Chunk size for memory-efficient processing (default: 1000)')

    return parser.parse_args()


def main() -> None:
    """
    Main function to split afterburner catalog into healpix-based catalogs.
    
    This function:
    1. Reads the input QSO catalog
    2. Calculates healpix indices for all QSOs
    3. Groups QSOs by healpix
    4. Creates individual healpix-based redshift catalogs
    5. Processes healpix in parallel batches for performance
    """
    args = parse_arguments()
    
    # Check the QSO catalog exists
    qsocat_path = Path(args.qsocat)
    if not qsocat_path.exists():
        print(f"Error: cannot find {qsocat_path}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Reading QSO catalog: {qsocat_path}")
    
    # Read QSO catalog using fitsio for better performance
    try:
        qcat = fitsio.read(str(qsocat_path))
    except Exception as e:
        print(f"Error reading QSO catalog {qsocat_path}: {e}")
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
        print(f"Found {len(healpixels)} entries with {len(healpixlist)} unique healpix")
    
    # Setup output directory and logging
    altzdir_path = Path(args.altzdir)
    altzdir_path.mkdir(parents=True, exist_ok=True)
    
    logfile = setup_logging(args)
    
    # Process healpix in parallel batches
    if args.nproc > 1 and len(healpixlist) > 1:
        # Split healpix into batches for parallel processing
        batch_size = max(1, len(healpixlist) // (args.nproc * 4))  # Ensure enough batches for all processes
        healpix_batches = [healpixlist[i:i+batch_size] for i in range(0, len(healpixlist), batch_size)]
        
        if args.verbose:
            print(f"Processing {len(healpixlist)} healpix in {len(healpix_batches)} batches using {args.nproc} processes")
        
        results = []
        with ProcessPoolExecutor(max_workers=args.nproc) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(process_healpix_batch, batch, qcat, healpixels, args): batch 
                for batch in healpix_batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    
                    if args.verbose:
                        batch = future_to_batch[future]
                        print(f"Completed batch with {len(batch)} healpix")
                except Exception as e:
                    batch = future_to_batch[future]
                    print(f"Error processing batch with {len(batch)} healpix: {e}")
    else:
        # Sequential processing for small datasets or single process
        results = process_healpix_batch(healpixlist, qcat, healpixels, args)
    
    # Write results to log file
    with open(logfile, 'a') as f:
        for healpix, nqsos in results:
            if args.mock:
                zfilename = f"{args.zfileroot}-16-{healpix}.fits"
            else:
                zfilename = f"{args.zfileroot}-{args.survey}-{args.moon}-{healpix}.fits"
            f.write(f"{zfilename}: {nqsos}\n")
    
    total_qsos = sum(nqsos for _, nqsos in results)
    if args.verbose:
        print(f"Finished processing {len(results)} healpix files with {total_qsos} total QSOs")
    
    print("Finished")


if __name__ == "__main__":
    main()
