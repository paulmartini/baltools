#!/usr/bin/env python

'''
Generate BAL catalogs from DESI healpix data for a specific data release. One 
catalog is generated per healpix. The catalogs are put in a directory 
structure that matches the structure of the data release. 
Use the separate script buildbalcat.py to create a BAL catalog for a 
corresponding QSO catalog. 

OPTIMIZED VERSION for NERSC systems with high core counts
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob
import fitsio

from time import gmtime, strftime
import argparse
from collections import defaultdict
import desispec.io
from desispec.coaddition import coadd_cameras
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import baltools
from baltools import balconfig as bc
from baltools import plotter, fitbal, baltable
from baltools import desibal as db
from baltools import utils

from multiprocessing import Pool

os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux'

class FileCache:
    """Cache for file existence checks to reduce I/O overhead"""
    def __init__(self):
        self._cache = {}
    
    def exists(self, filepath):
        if filepath not in self._cache:
            self._cache[filepath] = os.path.isfile(filepath)
        return self._cache[filepath]
    
    def clear(self):
        self._cache.clear()

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

    parser.add_argument('--mock', default=False, required = False, action='store_true',
                    help = 'Mock catalog?, default is False')

    parser.add_argument('--mockdir', type=str, default = None, required=False,
                    help='Path to directory structure with mock data (not including spectra-16/)') 

    parser.add_argument('-o','--outdir', type = str, default = None, required = False,
                    help = 'Deprecated -- now ignored')

    parser.add_argument('-l','--logfile', type = str, default = 'logfile.txt', required = False,
                    help = 'Name of log file written to altzdir, default is logfile.txt')

    parser.add_argument('--nproc', type=int, default=64, required=False,
                        help='Number of processes')

    parser.add_argument('-c','--clobber', default=False, required=False, action='store_true',
                    help='Clobber (overwrite) BAL catalog if it already exists?')

    parser.add_argument('-v','--verbose', default=False, required=False, action='store_true',
                    help = 'Provide verbose output?')

    parser.add_argument('-t','--alttemp', default=False, required=False, action='store_true',
                    help = 'Use alternate components made by Allyson Brodzeller')

    parser.add_argument('--tids', default=False, required=False, action='store_true',
                    help = 'Read only QSO TARGETIDs') 

    parser.add_argument('--chunk-size', type=int, default=50, required=False,
                        help='Chunk size for parallel processing (default: 50)')

    parser.add_argument('--file-discovery-workers', type=int, default=8, required=False,
                        help='Number of workers for parallel file discovery (default: 8)')

    if options is None: 
        args  = parser.parse_args()
    else: 
        args  = parser.parse_args(options)

    return args 

def discover_healpix_parallel(dataroot, n_workers=8):
    """Discover healpix directories in parallel"""
    def scan_directory(subdir):
        healpix_dirs = []
        if os.path.isdir(subdir):
            for item in os.listdir(subdir):
                full_path = os.path.join(subdir, item)
                if os.path.isdir(full_path):
                    healpix_dirs.append(item)
        return healpix_dirs
    
    # Get all subdirectories
    try:
        subdirs = [os.path.join(dataroot, d) for d in os.listdir(dataroot) 
                   if os.path.isdir(os.path.join(dataroot, d))]
    except FileNotFoundError:
        print(f"Warning: Data root directory {dataroot} not found")
        return []
    
    if not subdirs:
        return []
    
    healpixels = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all directory scans
        future_to_subdir = {executor.submit(scan_directory, subdir): subdir for subdir in subdirs}
        
        # Collect results as they complete
        for future in as_completed(future_to_subdir):
            try:
                result = future.result()
                healpixels.extend(result)
            except Exception as e:
                subdir = future_to_subdir[future]
                print(f"Error scanning directory {subdir}: {e}")
    
    return healpixels

def process_healpix_chunk(healpix_chunk, args, file_cache=None):
    """Process a chunk of healpix pixels"""
    if file_cache is None:
        file_cache = FileCache()
    
    results = []
    for healpix in healpix_chunk:
        result = findbals_one_healpix_optimized(healpix, args, file_cache)
        results.append(result)
    
    return results

def findbals_one_healpix_optimized(healpix, args, file_cache=None):
    """Optimized version of findbals_one_healpix with caching and early exits"""
    if file_cache is None:
        file_cache = FileCache()
    
    skiphealpix = False
    hpdir = utils.gethpdir(healpix) 
    
    if args.mock: 
        coaddfilename = f"spectra-16-{healpix}.fits"
        balfilename = coaddfilename.replace('spectra-16-', 'baltable-16-')
        altzdir = os.path.join(args.altzdir, 'spectra-16', hpdir, healpix) 
    else: 
        coaddfilename = f"coadd-{args.survey}-{args.moon}-{healpix}.fits"
        balfilename = coaddfilename.replace('coadd-', 'baltable-')
        altzdir = os.path.join(args.altzdir, "healpix", args.survey, args.moon, hpdir, healpix) 

    zfilename = balfilename.replace('baltable-', args.zfileroot+"-")
    indir = os.path.join(args.dataroot, hpdir, healpix)
    outdir = os.path.join(args.outroot, hpdir, healpix)

    coaddfile = os.path.join(indir, coaddfilename) 
    balfile = os.path.join(outdir, balfilename) 
    zfile = balfile.replace('baltable-', args.zfileroot+"-")

    # Early exit if output exists and not clobbering
    if file_cache.exists(balfile) and not args.clobber:
        return healpix, None

    # Check to see if zfile exists -- if not, skip
    if not file_cache.exists(zfile): 
        skiphealpix = True

    if args.verbose:
        print(f"Coadd file: {coaddfile}")
        print(f"BAL file: {balfile}")
        if args.altzdir is not None: 
            print(f"Redshift directory: {altzdir}")
        if skiphealpix: 
            print(f"Did not find {zfile}, so skipping healpix {healpix}")

    errortype = None
    if not file_cache.exists(balfile) or args.clobber:
        try:
            if not skiphealpix: 
                if args.verbose:
                    print(f"About to run db.desibalfinder with verbose={args.verbose} and altbaldir={outdir} and zfileroot={args.zfileroot}")
                db.desibalfinder(
                    coaddfile, 
                    altbaldir=outdir, 
                    altzdir=altzdir, 
                    zfileroot=args.zfileroot, 
                    overwrite=args.clobber, 
                    verbose=args.verbose, 
                    release=args.release, 
                    usetid=args.tids, 
                    alttemp=args.alttemp
                )
            else: 
                errortype = f"Did not find redshift catalog {zfile}"
        except Exception as e:
            if args.verbose:
                print(f"An error occurred at healpix {healpix}: {e}")
            errortype = str(e)

    return healpix, errortype

def main(args=None): 

    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    # Root directory for input data: 
    if args.mock: 
        dataroot = os.path.join(args.mockdir, 'spectra-16') 
    else: 
        dataroot = os.path.join(os.getenv("DESI_SPECTRO_REDUX"), args.release, "healpix", args.survey, args.moon) 
        
    # Root directory for output individual BAL catalogs: 
    if args.mock: 
        outroot = os.path.join(args.altzdir, 'spectra-16') 
    else: 
        outroot = os.path.join(args.altzdir, "healpix", args.survey, args.moon)
    utils.pmmkdir(outroot)

    if args.outdir is not None:
        print(f"Warning: --outdir is deprecated. Using {outroot} based on altzdir parameter") 
    
    # Discover healpix directories in parallel
    if args.verbose:
        print(f"Discovering healpix directories in {dataroot} using {args.file_discovery_workers} workers")
    
    healpixels = discover_healpix_parallel(dataroot, n_workers=args.file_discovery_workers)
    
    if args.verbose:
        print(f"Found {len(healpixels)} healpix directories")
    
    # Requested healpix
    inputhealpixels = args.healpix
    
    # Check that all requested healpix exist
    if inputhealpixels is not None:
        for inputhealpixel in inputhealpixels: 
            if str(inputhealpixel) not in healpixels:
                print(f"Warning: Healpix {inputhealpixel} not available in {dataroot}")
        # Filter to only available healpix
        inputhealpixels = [h for h in inputhealpixels if str(h) in healpixels]
    else:
        inputhealpixels = healpixels
    
    if not inputhealpixels:
        print("No healpix to process!")
        return
    
    # Create/confirm output healpix directories exist
    if args.verbose:
        print(f"Creating output directories for {len(inputhealpixels)} healpix")
    
    for inputhealpixel in inputhealpixels: 
        hpdir = utils.gethpdir(inputhealpixel)
        healpixdir = os.path.join(outroot, hpdir, inputhealpixel) 
        utils.pmmkdir(healpixdir) 
    
    # List of healpix that caused issues for by hand rerun.
    issuehealpixels = []
    errortypes = []
    
    outlog = os.path.join(outroot, args.logfile)
    
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
                print("Error with tagging log file") 
                lastupdate = f"Last updated {strftime('%Y-%m-%d %H:%M:%S', gmtime())} UT\n"
    
    with open(outlog, 'a') as f:
        f.write(lastupdate)
        f.write(" ".join(sys.argv) + '\n')

    # Add dataroot and outroot to args for use in worker functions
    args.dataroot = dataroot
    args.outroot = outroot

    # Process healpix in parallel chunks
    if args.nproc > 1 and len(inputhealpixels) > 1:
        # Split healpix into chunks for parallel processing
        chunk_size = max(1, min(args.chunk_size, len(inputhealpixels) // (args.nproc * 2)))
        healpix_chunks = [inputhealpixels[i:i+chunk_size] for i in range(0, len(inputhealpixels), chunk_size)]
        
        if args.verbose:
            print(f"Processing {len(inputhealpixels)} healpix in {len(healpix_chunks)} chunks using {args.nproc} processes")
        
        all_results = []
        with ProcessPoolExecutor(max_workers=args.nproc) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(process_healpix_chunk, chunk, args): chunk 
                for chunk in healpix_chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    
                    if args.verbose:
                        chunk = future_to_chunk[future]
                        print(f"Completed chunk with {len(chunk)} healpix")
                except Exception as e:
                    chunk = future_to_chunk[future]
                    print(f"Error processing chunk with {len(chunk)} healpix: {e}")
                    # Add error results for this chunk
                    for healpix in chunk:
                        all_results.append((healpix, str(e)))
    else: 
        # Sequential processing
        all_results = []
        for healpix in inputhealpixels:
            result = findbals_one_healpix_optimized(healpix, args)
            all_results.append(result)

    # Process results
    for healpix, errortype in all_results:
        if errortype is not None: 
            issuehealpixels.append(healpix) 
            errortypes.append(errortype) 

    # Write error summary to log
    with open(outlog, 'a') as f:
        f.write("List of healpix with errors and error types: \n")
        for i in range(len(issuehealpixels)):
            f.write(f"{issuehealpixels[i]} : {errortypes[i]}\n")
    
    if args.verbose:
        print(f"Wrote output log {outlog}")
        print(f"Processed {len(all_results)} healpix, {len(issuehealpixels)} had errors")

def _func(arg): 
    """ Used for multiprocessing.Pool - kept for backward compatibility """
    return findbals_one_healpix_optimized(**arg)

def findbals_one_healpix(healpix, args, healpixels, dataroot, outroot): 
    """Original function kept for backward compatibility"""
    # Create a temporary args object with dataroot and outroot
    temp_args = argparse.Namespace(**vars(args))
    temp_args.dataroot = dataroot
    temp_args.outroot = outroot
    
    return findbals_one_healpix_optimized(healpix, temp_args)

if __name__ == "__main__":
    main()

