#!/usr/bin/env python
"""

baltools.splitafterburner_hp
=========================

Split an afterburner redshift catalog into separate, healpix-based redshift catalogs in order 
to run the balfilder on individual healpix files

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
from baltools import popqsotab as pt
from baltools import utils

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

parser.add_argument('--mock', default=False, required = False, action='store_true',
                    help = 'Mock catalog?, default is False')

parser.add_argument('-l','--logfile', type = str, default = 'logfile-{survey}-{moon}.txt', required = False,
                    help = 'Name of log file written to outdir, default is logfile-{survey}-{moon}.txt')

parser.add_argument('-c','--clobber', default=False, required=False, action='store_true', 
                    help='Clobber (overwrite) BAL catalog if it already exists?')

parser.add_argument('-v','--verbose', default=False, required=False, action='store_true', 
                    help = 'Provide verbose output?')

parser.add_argument('--nproc', type=int, default=64, required=False,
                    help='Number of processes for parallel processing (default: 64)')

parser.add_argument('--chunk-size', type=int, default=1000, required=False,
                    help='Chunk size for memory-efficient processing (default: 1000)')

args  = parser.parse_args()

def calculate_healpix_vectorized(ra, dec, nside=64):
    """Calculate healpix for all coordinates at once using vectorized operations"""
    return hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)

def create_healpix_data(qcat, healpixels, hmask, args):
    """Create healpix data structure for a single healpix"""
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

def write_healpix_file(healpix, data, args):
    """Write a single healpix file using fitsio for better performance"""
    hpdir = utils.gethpdir(str(healpix))
    
    if args.mock:
        zfilename = f"{args.zfileroot}-16-{healpix}.fits"
        zdir = os.path.join(args.altzdir, "spectra-16", hpdir, str(healpix))
    else:
        zfilename = f"{args.zfileroot}-{args.survey}-{args.moon}-{healpix}.fits"
        zdir = os.path.join(args.altzdir, "healpix", args.survey, args.moon, hpdir, str(healpix))
    
    utils.pmmkdir(zdir)
    zfile = os.path.join(zdir, zfilename)
    
    # Check if file exists and clobber setting
    if os.path.isfile(zfile) and not args.clobber:
        if args.verbose:
            print(f"Warning: {zfile} exists and clobber = False, skipping")
        return healpix, 0
    
    # Write using fitsio for better performance
    fitsio.write(zfile, data, extname='REDSHIFTS', clobber=args.clobber)
    
    if args.verbose:
        print(f"Wrote output file {zfile} with {len(data)} QSOs")
    
    return healpix, len(data)

def process_healpix_batch(healpix_batch, qcat, healpixels, args):
    """Process a batch of healpix pixels"""
    results = []
    for healpix in healpix_batch:
        hmask = healpixels == healpix
        if np.sum(hmask) > 0:
            data = create_healpix_data(qcat, healpixels, hmask, args)
            healpix_result, nqsos = write_healpix_file(healpix, data, args)
            results.append((healpix_result, nqsos))
    return results

def main():
    # Check the QSO catalog exists
    if not os.path.isfile(args.qsocat):
        print("Error: cannot find ", args.qsocat)
        exit(1)
    
    if args.verbose:
        print(f"Reading QSO catalog: {args.qsocat}")
    
    # Read QSO catalog using fitsio for better performance
    qcat = fitsio.read(args.qsocat)
    
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
    if not os.path.isdir(args.altzdir):
        utils.pmmkdir(args.altzdir)
    
    if args.mock:
        logfile = os.path.join(args.altzdir, "logfile-mock.txt")
    else:
        logfile = os.path.join(args.altzdir, f"logfile-{args.survey}-{args.moon}.txt")
    
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
                batch_results = future.result()
                results.extend(batch_results)
                
                if args.verbose:
                    batch = future_to_batch[future]
                    print(f"Completed batch with {len(batch)} healpix")
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
