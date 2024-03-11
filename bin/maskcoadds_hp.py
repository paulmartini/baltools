#!/usr/bin/env python

'''
Mask the locations of the BAL features in the DESI healpix data for a specific data release. 
This is to mask the spectral regions contaminated by BAL absorption in order to rerun redrock. 
The script creates one new coadd per healpix. The catalogs are put in a directory 
structure that matches the structure of the data release. 
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.constants import c
import healpy as hp
from glob import glob

from time import gmtime, strftime
import argparse
import desispec.io

import baltools
from baltools import utils

from multiprocessing import Pool

os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux'


def parse(options=None): 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Create coadds with BALs masked""")

    parser.add_argument('-hp','--healpix', nargs='*', default = None, required=False,
                    help='List of healpix number(s) to process - default is all')

    parser.add_argument('-r', '--release', type = str, default = 'iron', required = False,
                    help = 'Data release subdirectory, default is iron')

    parser.add_argument('-w','--workdir', type=str, default = None, required=True,
                    help='Path to working directory structure for masked coadds')

    parser.add_argument('-q','--qsocat', type=str, default=None, required=True,
                    help='Full path to QSO (BAL) catalog') 

    parser.add_argument('-s', '--survey', type = str, default = 'main', required = False,
                    help = 'Survey subdirectory [sv1, sv2, sv3, main], default is main')

    parser.add_argument('-m', '--moon', type = str, default = 'dark', required = False,
                    help = 'Moon brightness [bright, dark], default is dark')

    parser.add_argument('--mock', default=False, required = False, action='store_true',
                    help = 'Mock catalog?, default is False')

    parser.add_argument('--mockdir', type=str, default = None, required=False,
                    help='Path to directory structure with mock data (not including spectra-16/)') 

    parser.add_argument('-l','--logfile', type = str, default = 'logfile.txt', required = False,
                    help = 'Name of log file written to workdir, default is logfile.txt')

    parser.add_argument('--nproc', type=int, default=128, required=False,
                        help='Number of processes')

    parser.add_argument('-c','--clobber', default=False, required=False, action='store_true',
                    help='Clobber (overwrite) output coadd if it already exists?')

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

    # Check the QSO catalog exists
    if not os.path.isfile(args.qsocat):
        print("Error: cannot find ", args.qsocat)
        exit(1)

    qcat = fits.open(args.qsocat)[1].data

    # Root directory for input data: 
    if args.mock: 
        dataroot = os.path.join(args.mockdir, 'spectra-16') 
    else: 
        dataroot = os.path.join(os.getenv("DESI_SPECTRO_REDUX"), args.release, "healpix", args.survey, args.moon) 
        
    # Root directory for output coadds: 
    if args.mock: 
        outroot = os.path.join(args.workdir, 'spectra-16') 
    else: 
        outroot = os.path.join(args.workdir, "healpix", args.survey, args.moon)
    utils.pmmkdir(outroot)

    # Calculate healpix for every QSO
    if args.mock:
        healpixels = hp.ang2pix(16, qcat['RA'], qcat['DEC'], lonlat=True, nest=True)
    else:
        healpixels = hp.ang2pix(64, qcat['TARGET_RA'], qcat['TARGET_DEC'], lonlat=True, nest=True)

    # Construct a list of unique healpix pixels
    healpixlist = np.unique(healpixels)

    print(f"Found {len(healpixlist)} unique healpixels")
    
    # Requested healpix
    inputhealpixels = np.asarray(args.healpix)
    
    workdir = None
    
    # Check that all requested healpix exist
    if args.healpix is not None:
        print("inputhealpixels = ", inputhealpixels) 
        for inputhealpixel in inputhealpixels: 
            print(f"inputhealpixel = {inputhealpixel}")
            assert(int(inputhealpixel) in healpixlist), "Healpix {} not available".format(inputhealpixel)
    # If there were no inputhealpixels, process all of the unique list
    else:
        inputhealpixels = np.asarray(healpixlist)
    
    print(f"Length of inputhealpixels = {len(inputhealpixels)}") 

    # Create/confirm output healpix directories exist
    for inputhealpixel in inputhealpixels: 
        hpdir = utils.gethpdir(str(inputhealpixel))
        healpixdir = os.path.join(outroot, hpdir, str(inputhealpixel)) 
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
        func_args = [ {"healpix": healpix, \
                       "args": args, \
                       "healpixels": healpixels, \
                       "dataroot": dataroot, \
                       "outroot": outroot, \
                       "qcat": qcat 
                    } for ih,healpix in enumerate(inputhealpixels) ]
        with Pool(args.nproc) as pool: 
            results = pool.map(_func, func_args)
    else: 
        for ih,healpix in enumerate(inputhealpixels) : 
            results = maskbals_one_healpix(str(healpix), args, healpixels, dataroot, outroot, qcat) 

    lresults = list(results) 
    print('lresults = ', lresults) 

#    if len(lresults) == 2: 
#        if lresults[1] is not None: 
#            issuehealpixels.append(lresults[i][0]) 
#            errortypes.append(lresults[i][1]) 
#    else: 
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
    return maskbals_one_healpix(**arg)


def maskbals_one_healpix(healpix, args, healpixels, dataroot, outroot, qcat): 
    """
    healpix: unique healpix to process
    healpixels: healpix for each quasar in the catalog
    1) Get the BAL targetids in the healpix 
    2) Open the coadd file
    3) Mask the BAL troughs in LyA, NV, SiIV, and CIV
    4) Write a new coadd file
    """

    skiphealpix = False
    if not isinstance(healpix, str):
        healpix = str(healpix)
    hpdir = utils.gethpdir(healpix) 
    if args.mock: 
        coaddfilename = "spectra-16-{0}.fits".format(healpix) 
        workdir = os.path.join(args.workdir, 'spectra-16', hpdir, healpix) 
    else: 
        coaddfilename = "coadd-{0}-{1}-{2}.fits".format(args.survey, args.moon, healpix) 
        workdir = os.path.join(args.workdir, 'healpix', args.survey, args.moon, hpdir, healpix) 

    indir = os.path.join(dataroot, hpdir, healpix)
    outdir = os.path.join(outroot, hpdir, healpix)

    coaddfile = os.path.join(indir, coaddfilename) 
    outcoaddfile = os.path.join(outdir, coaddfilename) 

    if args.verbose:
        print("Coadd file: ", coaddfile)
        if args.workdir is not None: 
            print("Working (scratch) directory: ", workdir)
        if skiphealpix: 
            print("Did not find {0}, so skipping healpix {1}".format(zfile, healpix))

    print(f"healpix = ", healpix)
    print(f"healpixels = ", healpixels[:20])
    print(f"len(healpixels) = {len(healpixels)}")
    hindxs = np.arange(0, len(qcat), dtype=int)
    hmask = healpixels == int(healpix) # mask for the current healpix
    print(f"hmask = ", hmask[:20])
    # print(f"len(hmask) = {len(hmask)}")
    print(f"np.sum(hmask) = {np.sum(hmask)}") 
    bmask = qcat['AI_CIV'] > 0 # mask for AI_CIV > 0 BALs
    qindxs = hindxs[hmask*bmask] # indices in qcat that are BALs in this healpix
    print(f"len(qindxs) = {len(qindxs)}")
    print(f"qindxs = ", qindxs)
    print(f"Healpix {healpix} has {np.sum(hmask*bmask)} BALs in {np.sum(hmask)} QSOs") 

    errortype = None
    if not os.path.isfile(outcoaddfile) or args.clobber:
        try:
            if not skiphealpix: 
                print(f"About to create masked coadd with verbose={args.verbose} and outdir={outdir} and outcoaddfile={outcoaddfile}") 
                # Read in the coadd
                spectra = desispec.io.read_spectra(coaddfile, single=True)
                print("Read in coadd file", coaddfile)
                # Loop through indices in the QSO catalog
                for qindx in qindxs:
                    targetid = qcat['TARGETID'][qindx]
                    # find the index of this targetid in the coadd (cindx) 
                    cindx =  np.where(targetid == spectra.fibermap['TARGETID'])[0][0]
                    print(f"Masking BAL {targetid} {spectra.fibermap['TARGETID'][cindx]} with AI_CIV = {qcat['AI_CIV'][qindx]:.0f}") 
                    # If AI_CIV > 0, identify BAL velocity limits and mask them:
                    if qcat['NCIV_450'][qindx] > 0.:
                        zspec = qcat['Z'][qindx]
                        if args.verbose: 
                            print(f"BAL {qcat['TARGETID'][qindx]} at index: {cindx} with NCIV_450: {qcat['NCIV_450'][qindx]} AI_CIV: {qcat['AI_CIV'][qindx]:.2f} zspec =  {zspec:.3f}") 
                        ### Mask CIV
                        for t in range(qcat['NCIV_450'][qindx]): # loop over troughs
                            vmin = qcat['VMIN_CIV_450'][qindx][t]
                            vmax = qcat['VMAX_CIV_450'][qindx][t]
                            # Calculate the corresponding wavelengths
                            lmax = 1549*(1. - vmin/c.to('km/s').value)*(1.+zspec)
                            lmin = 1549*(1. - vmax/c.to('km/s').value)*(1.+zspec)
                            if args.verbose: 
                                print(f" (vmin, vmax) = ({qcat['VMIN_CIV_450'][qindx][t]:0f}, {qcat['VMAX_CIV_450'][qindx][t]:0f}), (lmin, lmax) = ({lmin:.1f}, {lmax:.1f}))") 
                            mask_trough(spectra, lmin, lmax, cindx, verbose=args.verbose)
                        ## SiIV
                        for t in range(qcat['NSIIV_450'][qindx]): # loop over troughs
                            vmin = qcat['VMIN_SIIV_450'][qindx][t]
                            vmax = qcat['VMAX_SIIV_450'][qindx][t]
                            # Calculate the corresponding wavelengths
                            lmax = 1400*(1. - vmin/c.to('km/s').value)*(1.+zspec)
                            lmin = 1400*(1. - vmax/c.to('km/s').value)*(1.+zspec)
                            mask_trough(spectra, lmin, lmax, cindx, verbose=args.verbose)
                        ### Mask NV
                        for t in range(qcat['NCIV_450'][qindx]): # loop over troughs
                            vmin = qcat['VMIN_CIV_450'][qindx][t]
                            vmax = qcat['VMAX_CIV_450'][qindx][t]
                            # Calculate the corresponding wavelengths
                            lmax = 1241*(1. - vmin/c.to('km/s').value)*(1.+zspec)
                            lmin = 1241*(1. - vmax/c.to('km/s').value)*(1.+zspec)
                            mask_trough(spectra, lmin, lmax, cindx, verbose=args.verbose)
                        ### Mask LyA
                        for t in range(qcat['NCIV_450'][qindx]): # loop over troughs
                            vmin = qcat['VMIN_CIV_450'][qindx][t]
                            vmax = qcat['VMAX_CIV_450'][qindx][t]
                            # Calculate the corresponding wavelengths
                            lmax = 1216*(1. - vmin/c.to('km/s').value)*(1.+zspec)
                            lmin = 1216*(1. - vmax/c.to('km/s').value)*(1.+zspec)
                            mask_trough(spectra, lmin, lmax, cindx, verbose=args.verbose)
                # Write output:
                spectra.scores_comments = None  # Not sure why I need this
                outfile = desispec.io.write_spectra(outcoaddfile,  spectra)
                print(f"Wrote masked coadd {outcoaddfile}")
            else: 
                errortype = "Did not find spectrum {0}".format(coaddfile)
                # issuehealpixels.append(healpix)
                # errortypes.append(errortype)
        except:
            print("An error occured at healpix {}. Adding healpix to issuehealpixels list.".format(healpix))
            errortype = sys.exc_info()[0]


    return healpix, errortype



def mask_trough(spec, lmin, lmax, ind, verbose=False):
    '''
    Mask BAL troughs from lmin to lmax for a spectrum with index ind by setting ivar = 0
    
    Parameters
    ----------
    spec : Spectrum object
        coadd spectrum
    lmin, lmax : float
        first and last wavelength of trough
    ind : int
        index of BAL to modify

    Returns
    -------
    none
    
    '''

    if verbose: 
        print("Masking troughs in", spec.fibermap['TARGETID'][ind], "index = ", ind, "in coadd file")

    bands = ['b', 'r', 'z']
    for band in bands:
        bandmin = spec.wave[band][0]
        bandmax = spec.wave[band][-1]
        if verbose: 
            print("{0}: {1:.2f} {2:.2f}".format(band, bandmin, bandmax))
        if lmin <= bandmin:
            # Started on previous band
            i1 = 0
            if lmax >= bandmax: 
                i2 = -1
                spec.ivar[band][ind][0:i2] = 0.
                if verbose:  
                    print("Masked {0:.0f} to {1:.0f} in {2}".format(spec.wave[band][i1], spec.wave[band][i2], band))
                    print(spec.ivar[band][ind][0:i2])
            elif lmax >= bandmin and lmax <= bandmax:
                # Finishes in this band
                i2 = np.where(spec.wave[band] <= lmax)[0][-1]
                spec.ivar[band][ind][0:i2] = 0.
                if verbose:
                    print("Masked {0:.0f} to {1:.0f} in {2}".format(spec.wave[band][i1], spec.wave[band][i2], band))
                    print(spec.ivar[band][ind][0:i2])
            else:
                if verbose:
                    print("At bluer wavelengths than", band)
        elif lmin >= bandmin and lmin <= bandmax:
            # Starts in this band
            i1 = np.where(spec.wave[band] >= lmin)[0][0]
            if lmax <= bandmax:
                # Ends in this band
                i2 = np.where(spec.wave[band] <= lmax)[0][-1]
                spec.ivar[band][ind][i1:i2] = 0.
                if verbose: 
                    print("Masked {0:.0f} to {1:.0f} in {2}".format(spec.wave[band][i1], spec.wave[band][i2], band))
                    print(spec.ivar[band][ind][i1:i2])
            elif lmax > bandmax:
                i2 = -1
                spec.ivar[band][ind][i1:i2] = 0.
                if verbose: 
                    print("Masked {0:.0f} to {1:.0f} in {2}".format(spec.wave[band][i1], spec.wave[band][i2], band))
                    print(spec.ivar[band][ind][i1:i2])
        else:
            if verbose:
                print("At redder wavelengths than", band)


if __name__ == "__main__":
    main()

