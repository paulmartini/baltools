#!/usr/bin/env python3
"""
Parallelized appendbalinfo_hp: efficiently updates a QSO catalog with BAL info from per-healpix catalogs.

Each worker processes a distinct chunk of healpix, reads only relevant rows from the QSO catalog,
matches to BALCATs by TARGETID, and returns only the updates. 

Parallel refactor by Copilot (2025)
"""

import os
import sys
import numpy as np
import fitsio
import healpy as hp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from time import gmtime, strftime
import argparse

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
        if balcol in binfo.dtype.names: 
            qinfo[balcol] = binfo[balcol]
    
    qinfo['BALMASK'] = 0


def calculate_healpix(qcat, mock):
    if mock:
        return hp.ang2pix(16, qcat['RA'], qcat['DEC'], lonlat=True, nest=True)
    else:
        return hp.ang2pix(64, qcat['TARGET_RA'], qcat['TARGET_DEC'], lonlat=True, nest=True)


def match_targets(qso_targetids, bal_targetids):
    # Return list of (qso_index, bal_index)
    qso_map = {tid: idx for idx, tid in enumerate(qso_targetids)}
    matches = []
    for bal_idx, bal_tid in enumerate(bal_targetids):
        if bal_tid in qso_map:
            matches.append((qso_map[bal_tid], bal_idx))
    return matches


def process_healpix_chunk(
    healpix_chunk,
    qsocat_path,
    baldir,
    survey,
    moon,
    mock,
    alttemp,
    verbose
):
    """
    Each worker:
    - loads QSO catalog,
    - selects only rows for its assigned healpix,
    - loads BALCATs for those healpix,
    - matches TARGETIDs,
    - returns list of (qso_index, bal_data) tuples (global indices in the QSO catalog).
    """
    import fitsio
    import numpy as np
    import healpy as hp
    from baltools import utils

    modifications = []

    # Load QSO catalog
    qcat = fitsio.read(str(qsocat_path), ext=1)
    healpixels = calculate_healpix(qcat, mock)
    hindxs = np.arange(0, len(qcat), dtype=int)

    for healpix in healpix_chunk:
        hpdir = utils.gethpdir(str(healpix))
        if mock:
            balfilename = f"baltable-16-{healpix}.fits"
            balfile = Path(baldir) / 'spectra-16' / hpdir / str(healpix) / balfilename
        else:
            balfilename = f"baltable-{survey}-{moon}-{healpix}.fits"
            balfile = Path(baldir) / "healpix" / survey / moon / hpdir / str(healpix) / balfilename
        try:
            bcat = fitsio.read(str(balfile), ext='BALCAT')
        except (FileNotFoundError, OSError):
            if verbose:
                print(f"Warning: Did not find {balfile}")
            continue
        # bcat = fitsio.read(str(balfile), ext='BALCAT')


        hmask = healpixels == healpix
        indxs = hindxs[hmask]
        targids = qcat['TARGETID'][hmask]

        matches = match_targets(targids, bcat['TARGETID'])
        for qso_idx_local, bal_idx in matches:
            qcat_index = indxs[qso_idx_local]
            # We must copy the bal_data here, since memory will be released after process exit
            bal_data_copy = bcat[bal_idx].copy()
            modifications.append((qcat_index, bal_data_copy))
        if verbose:
            print(f"Healpix {healpix}: {len(bcat)} BAL entries, {len(matches)} matched.")
    return modifications


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Update existing QSO catalogue with BAL information (parallel version)"
    )
    parser.add_argument('-q', '--qsocat', type=str, required=True, help='Input QSO catalog')
    parser.add_argument('-b', '--baldir', type=str, required=True, help='Path to directory structure with individual BAL catalogs')
    parser.add_argument('-o', '--outcatfile', type=str, default="qso-balcat.fits", help='Output QSO+BAL catalog')
    parser.add_argument('-s', '--survey', type=str, default='main', help='Survey subdirectory [sv1, sv2, sv3, main], default is main')
    parser.add_argument('-m', '--moon', type=str, default='dark', help='Moon brightness [bright, dark], default is dark')
    parser.add_argument('--mock', default=False, action='store_true', help='Mock catalog?, default is False')
    parser.add_argument('-l', '--logfile', type=str, default=None, help='Name of log file (auto by default)')
    parser.add_argument('-c', '--clobber', default=False, action='store_true', help='Clobber (overwrite) BAL catalog if it already exists?')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Provide verbose output?')
    parser.add_argument('-t', '--alttemp', default=False, action='store_true', help='Use alternate components made by Allyson Brodzeller')
    parser.add_argument('--nproc', type=int, default=64, help='Number of processes for parallel processing')
    parser.add_argument('--chunk-size', type=int, default=160, help='Healpix per worker chunk')
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux'

    qsocat_path = Path(args.qsocat)
    outcat = Path(args.outcatfile)

    if not qsocat_path.exists():
        print(f"Error: cannot find {qsocat_path}")
        sys.exit(1)

    # Add empty BAL columns to QSO catalog, write to outcat
    cols = pt.inittab(str(qsocat_path), str(outcat), alttemp=args.alttemp)
    qcat = fitsio.read(str(qsocat_path), ext=1)

    # Calculate healpix for every QSO
    healpixels = calculate_healpix(qcat, args.mock)
    healpixlist = np.unique(healpixels)

    if args.verbose:
        print(f"Loaded QSO catalog: {len(qcat)} entries, {len(healpixlist)} unique healpix")
        print(f"Processing with {args.nproc} processes, {args.chunk_size} healpix per worker")

    # Set up logging
    if args.logfile is not None:
        logfile = Path(args.logfile)
    else:
        if args.mock:
            logfile = Path(args.baldir) / f"logfile-mock.txt"
        else:
            logfile = Path(args.baldir) / f"logfile-{args.survey}-{args.moon}.txt"
    with open(logfile, 'a') as f:
        lastupdate = f"Last updated {strftime('%Y-%m-%d %H:%M:%S', gmtime())} UT by {os.getenv('USER', 'unknown')}\n"
        f.write(lastupdate)
        f.write(" ".join(sys.argv) + '\n')
        f.write("Healpix NQSOs Nmatches\n")

    # Parallel processing setup
    all_modifications = []
    healpix_chunks = [healpixlist[i:i+args.chunk_size] for i in range(0, len(healpixlist), args.chunk_size)]

    with ProcessPoolExecutor(max_workers=args.nproc) as executor:
        futures = [
            executor.submit(
                process_healpix_chunk,
                chunk,
                str(outcat),
                args.baldir,
                args.survey,
                args.moon,
                args.mock,
                args.alttemp,
                args.verbose
            ) for chunk in healpix_chunks
        ]
        for future in as_completed(futures):
            try:
                mods = future.result()
                all_modifications.extend(mods)
            except Exception as e:
                print(f"Error in worker: {e}")

    # Apply all modifications
    if args.verbose:
        print(f"Applying {len(all_modifications)} BAL property modifications to QSO catalog")
    for qcat_index, bal_data in all_modifications:
        balcopy(qcat[qcat_index], bal_data)

    # Write log for matches (optional: can log per healpix if desired)
    # Here, just total matches
    with open(logfile, 'a') as f:
        f.write(f"Total matches: {len(all_modifications)}\n")

    # Apply redshift range mask
    zmask = (qcat['Z'] < bc.BAL_ZMIN) | (qcat['Z'] > bc.BAL_ZMAX)
    zbit = 2 * np.ones(len(qcat), dtype=np.ubyte)
    qcat['BALMASK'][zmask] += zbit[zmask]

    # Write final catalog
    fitsio.write(str(outcat), qcat, extname='ZCATALOG', clobber=True)
    print(f"Wrote {outcat}")


if __name__ == "__main__":
    main() 
