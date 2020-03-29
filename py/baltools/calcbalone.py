#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:05:09 2019

@author: martini
"""

from astropy.io import fits
from astropy import constants as const
import fitbal
import balconfig as bc 
import plotter
import utils 
import fitsio

def balfitqso(sdssname="", pmf=[], showplot=True, savefig=True, outfig="fit.png", verbose=True, printbal=False, qsocat="DR14"): 
    '''
    Run BAL finder on a single QSO 

    Parameters
    ----------
    sdssname : string
        SDSS_NAME for QSO 
    pmf : 1-d int array
        PLATE/MJD/FIBERID for QSO 
    showplot : bool
        plot the fit
    savefig : bool
        save the figure?
    outfig : string
        name of the output figure
    verbose : bool
        turn on or off some progress reporting
    printbal : bool
        turn on or off bal output
    qsocat : string
        qso catalog (either "DR14" or "DR16")

    Returns
    -------
    balinfo : array
	array with BAL parameters 
    pcacoeffs : 1-d float array
	NPCA coefficients of PCA fit 
    calcmask : 1-d boolean array
	mask of BAL features 
    '''

    if qsocat == "DR14": 
        qsocathdu = fits.open(bc.qsodr14file)
    elif qsocat == "DR16":
        qsocathdu = fits.open(bc.qsodr16file)
    elif qsocat == "DR12":
        qsocathdu = fits.open(bc.qsodr12file)
    else: 
        raise RuntimeError("qsocat must be DR12, DR14, or DR16")

    # Determine if sdssname or plate info was specified, get more details

    if len(pmf) == 0 and len(sdssname) == 0: 
      raise RuntimeError("Specify either sdssnamne=SDSS_NAME or pmf = [plate, MJD, fiberid]")
  
    # if no SDSS_NAME is specified, try pmf 
    if len(sdssname) == 0: 
        try: 
            plate, mjd, fiberid = pmf 
            qindx = fitbal.qsocatsearch(qsocathdu[1].data, pmf=pmf)
        except ValueError: 
            raise ValueError("Need three values to unpack pmf")
    # otherwise try the SDSS_NAME
    else: 
        qindx = fitbal.qsocatsearch(qsocathdu[1].data, sdssname=sdssname)

    if len(sdssname) == 0: 
        sdssname = qsocathdu[1].data['SDSS_NAME'][qindx] 

    zpca = qsocathdu[1].data['Z_PCA'][qindx]
    print("balfitqso:", sdssname, zpca)
    if qsocat == "DR16":
        qsospec = utils.getdr16spectra(qsocathdu[1].data[qindx], verbose=verbose)
    else:
        qsospec = utils.getdr14spectra(qsocathdu[1].data[qindx], verbose=verbose)
    outfigname = sdssname + "-fit.png" 

    try: 
        balinfo, pcaout, balmask = runcalcbalone(qsospec, qsocathdu[1].data[qindx], showplot=showplot, sdssname=sdssname, savefig=savefig, outfig=outfigname, verbose=verbose)
    except RuntimeError:
        raise RuntimeError

    qsocathdu.close()

    if printbal: 
        print(balinfo)

def runcalcbalone(qsospec, array, showplot=False, sdssname="", savefig=False, outfig="fit.png", verbose=True): 
    '''
    Run PCA fit and BAL characterization on a single QSO spectrum

    Parameters
    ----------
    qsospec : np.array
        qso spectrum
    array : table row
        row qso catalog
    showplot : bool
        display a plot of the fit 
    sdssname : string
        SDSS_NAME of QSO
    savefig : bool
        save the figure?
    outfig : string
        name of the output figure
    verbose : bool
        turn on or off some progress reporting


    Returns
    -------
    balinfo : dict
	dictionary with BAL parameters 
    pcaout : 1-d float array
	array with PCA coefficients and chi2 values
    calcmask : 1-d boolean array
	mask of BAL features 
    '''

    zpca = array['Z_PCA']
    pcaeigen = fitsio.read(bc.pcaeigenfile)
    print("calcbalone says zpca = ", zpca)

    try: 
        balinfo, pcaout, mask = fitbal.calcbalparams(qsospec, pcaeigen, zpca)
    except RuntimeError:
        raise RuntimeError

    print(balinfo)

    pcafit = fitbal.createpcatemplate(pcaeigen, pcaout[:bc.NPCA])

    if showplot:
        plotter.plotbal(qsospec, pcafit, pcaeigen, balinfo, zpca, pcaout, lam1=1400, lam2=1600, sdssname=sdssname, savefig=savefig, outfig=outfig)

    return balinfo, pcaout, mask
