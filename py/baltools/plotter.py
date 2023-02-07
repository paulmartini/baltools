"""
  
baltools.plotter 
================

Various convenience plotting routines 

"""

import numpy as np
import os
import random
import matplotlib.pyplot as plt
from astropy.io import fits 
from astropy import constants as const
import fitsio
import desispec.io

from baltools import utils
from baltools import fitbal 
from baltools import baltable
from baltools import balconfig as bc

c = const.c.to('km/s').value         # speed of light in km/s

def plotsdssname(sdssname, lam1=1340, lam2=1680, verbose=False, plotvar=True):
    ''' 
    Plot the SDSS spectrum with name SDSS_NAME

    Parameters
    ----------
    sdssname : string
        SDSS_NAME of QSO 
    lam1 : float
        first wavelength to show (rest frame)
    lam2 : float
        last wavelength
    verbose : bool
        turn on or off some progress reporting
    plotvar : bool
        add variance to plot

    Returns
    -------
    none
    '''
    
    qsocathdu = fits.open(bc.qsodr14file)
    qindx = fitbal.qsocatsearch(qsocathdu[1].data, sdssname=sdssname)
    try: 
        qsospec = utils.getdr14spectra(qsocathdu[1].data[qindx], verbose=verbose)
    except FileNotFoundError:
        try: 
            qsospec = utils.getdr16spectra(qsocathdu[1].data[qindx], verbose=verbose)
        except: 
            plate4 = str(array['PLATE'])
            fiberid4 = str(array['FIBERID'])
            mjd = array['MJD']
            specfits = "spec-%s-%s-%s.fits" % (plate4, mjd, fiberid4)
            raise FileNotFoundError("Couldn't find file {0}".format(specfits))
    plotsdssspec(qsospec, qsocathdu[1].data[qindx]['Z'], sdssname=sdssname, lam1=lam1, lam2=lam2, plotvar=plotvar)


def plotsdssspec(hdu, zspec, sdssname="", lam1=1340, lam2=1680, plotvar=False):
    ''' 
    Plot the SDSS spectrum in hdu, which is expected to have 
    wavelength in hdu[1].data['loglam'] and flux in hdu[1].data['flux'] 

    Parameters
    ----------
    hdu : fits HDU
        SDSS spectrum HDU
    zspec : float
        redshift
    sdssname : string
        SDSS_NAME of QSO 
    lam1 : float
        first wavelength to show (rest frame)
    lam2 : float
        last wavelength
    plotvar : bool
        add variance to plot

    Returns
    -------
    none
    '''
  
    fig, ax = plt.subplots(1, figsize=(12,8))
    lam_z = np.power(10, hdu[1].data['loglam'])/(1.+zspec) 
    mm = lam_z < lam2
    mm = mm * lam_z > lam1
    ax.plot(np.power(10, hdu[1].data['loglam'][mm])/(1+zspec),
             hdu[1].data['flux'][mm], label="Data")
    ax.plot(lam_z, hdu[1].data['model'], 'r:', label="SDSS Model") 
    if plotvar: 
        ax.plot(np.power(10, hdu[1].data['loglam'][mm])/(1+zspec),
             np.sqrt(1./hdu[1].data['ivar'][mm]))
    ax.set_xlim(lam1, lam2)

    # add label
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    if len(sdssname) > 0: 
        x1 = xmin + 0.05*(xmax-xmin)
        y1 = ymin + 0.95*(ymax-ymin)
        ax.text(x1,y1,sdssname, fontsize=14) 

    ax.set_xlabel('Wavelength', fontsize=14)
    ax.set_ylabel('Flux', fontsize=14)
    ax.legend(loc='upper right', fontsize=14)
    
    plt.show() 


def plotbalfromcat(array, lam1=1260, lam2=1680, gmflag=False, verbose=False): 
    '''
    Plot a BAL based on the entry in the BAL catalog

    Parameters
    ----------
    array : row
        entry in balhdu
    lam1 : float
        first wavelength to show (rest frame)
    lam2 : float
        last wavelength
    gmflag : bool
        need to change a few things if G&M (2019) catalog 

    Returns
    -------
    none
    '''

    plate4 = str(array['PLATE'])
    fiberid4 = str(array['FIBERID'])
    mjd = array['MJD']
    plate4 = utils.zeropad(plate4)
    fiberid4 = utils.zeropad(fiberid4)
    specfits = "spec-%s-%s-%s.fits" % (plate4, mjd, fiberid4)
    # qsospec = fits.open(bc.specdir + plate4 + '/' + specfits) 
    try: 
        qsospec = utils.getdr16spectra(array, verbose=verbose)[1].data
    except FileNotFoundError:
        qsospec = utils.getdr14spectra(array, verbose=verbose)[1].data

    # qsospec = utils.getdr14spectra(array)
    pcaeigen = fitsio.read(bc.pcaeigenfile)
    zpca = array['Z_PCA']
    if zpca < bc.BAL_ZMIN and array['Z'] > bc.BAL_ZMIN: 
        zpca = array['Z']
        print(f"Warning: Z_PCA = {array['Z_PCA']} < {bc.BAL_ZMIN} so setting Z = {array['Z']}") 

    fig, ax = plt.subplots(1, figsize=(12,8))
    lam_z = np.power(10, qsospec['loglam'])/(1.+zpca)
    mm = lam_z < lam2
    mm = mm * lam_z > lam1
    #ax.plot(np.power(10, qsospec['loglam'][mm])/(1+zpca),
    #        qsospec['flux'][mm])
    ax.plot(np.power(10, qsospec['loglam'][mm])/(1+zpca),
            qsospec['flux'][mm])
    ax.plot(lam_z, qsospec['model'], 'r:', label="SDSS Model") 
    ax.set_xlim(lam1, lam2)

    try: 
        pcafit = fitbal.createpcatemplate(pcaeigen, array['PCA_COEFFS'])
        ax.plot(pcaeigen['WAVE'], pcafit, label="PCA Model")
        ax.plot(pcaeigen['WAVE'], 0.9*pcafit, 'g:')
    except KeyError:
        if verbose: 
            print("No PCA_COEFFS")
    ymin, ymax = ax.get_ylim()

    VMAX_CIV_2000 = array['VMAX_CIV_2000']
    VMIN_CIV_2000 = array['VMIN_CIV_2000']
    VMAX_CIV_450 = array['VMAX_CIV_450']
    VMIN_CIV_450 = array['VMIN_CIV_450']

    balinfo = baltable.cattobalinfo(array)

    drawtroughs(ax, balinfo, ymin, ymax)

    xlab = lam1 + 0.95*(lam2-lam1)
    ylab1 = ymin + 0.95*(ymax-ymin)
    ylab2 = ymin + 0.9*(ymax-ymin)
    ylab3 = ymin + 0.85*(ymax-ymin)
    ax.text(xlab, ylab1, specfits, ha='right')
    zlab2 = "{0} z = {1:.2f}".format(array['SDSS_NAME'], zpca)
    zlab3 = "AI = {0:.0f} BI = {1:.0f}".format(array['AI_CIV'], array['BI_CIV'])
    ax.text(xlab, ylab2, zlab2, ha='right')
    ax.text(xlab, ylab3, zlab3, ha='right')
    ax.set_xlabel('Wavelength', fontsize=14)
    ax.set_ylabel('Flux', fontsize=14)
    
    ax.legend(loc='lower right', fontsize=14)
    plt.xlim(lam1, lam2)
    plt.show()

def plotdesibal2(datadir, balcat, targetid, lam1=1340, lam2=1680): 
    '''
    Plot spectrum of a BAL with TARGETID. 
    Finds the coadd data for the BAL
    Wrapper for plotdesibal()

    Parameters
    ----------
    datadir : directory 
        root directory  of data release
    balcat : numpy recarray
        BAL catalog that corresponds to specobj
    targetid : int
        DESI TARGETID
    lam1 : float, optional
        first wavelength to plot (default is 1340)
    lam2 : float, optional
        last wavelength to plot (default is 1680)

    Returns
    -------
    none
    '''

    qindx = np.where(balcat['TARGETID'] == targetid)[0][0]
    tileid = str(balcat['TILEID'][qindx])
    night = str(balcat['NIGHT'][qindx])
    sp = str(balcat['PETAL_LOC'][qindx])
    coaddpath = os.path.join(datadir, 'tiles', tileid, night)
    coaddfile = coaddpath + '/coadd-' + sp + '-' + tileid + '-' + night + '.fits'
    specobj = desispec.io.read_spectra(coaddfile)
    
    plotdesibal(specobj, balcat, targetid, lam1, lam2)

def plotdesibal(specobj, balcat, targetid, lam1=1340, lam2=1680): 
    '''
    Plot spectrum of a BAL in specobj with TARGETID. 
    Wrapper for plotbal()
    Default is to plot data coadded across cameras, otherwise will just plot all three

    Parameters
    ----------
    specobj : spectrum object
        DESI spectrum object
    balcat : numpy recarray
        BAL catalog that corresponds to specobj
    targetid : int
        DESI TARGETID
    lam1 : float, optional
        first wavelength to plot (default is 1340)
    lam2 : float, optional
        last wavelength to plot (default is 1680)

    Returns
    -------
    none
    '''

    # find QSO index of TARGETID in specobj: 
    fm = specobj.fibermap
    qindx = np.where(fm['TARGETID'] == targetid)[0][0]

    # find QSO index of TARGETID in balhdu: 
    bindx = np.where(balcat['TARGETID'] == targetid)[0][0]
    zspec = balcat['Z'][bindx]

    # Read in the eigenspectra
    pcaeigen = fitsio.read(bc.pcaeigenfile)

    # Create the spectrum for the figure: 
    if 'brz' in specobj.wave.keys():
        qsospec = np.zeros(len(specobj.wave['brz']),dtype={'names':('wave', 'flux', 'ivar', 'model'), 'formats':('>f8', '>f8', '>f8', '>f8')})
        qsospec['wave'] = specobj.wave['brz']
        qsospec['flux'] = specobj.flux['brz'][qindx]
        qsospec['ivar'] = specobj.ivar['brz'][qindx]
        qsospec['model'] = np.zeros(len(specobj.wave['brz'])) # for SDSS compatibility
    else: 
        print("Warning: using non-coadded camera data")
        nwaves = len(specobj.wave['b']) + len(specobj.wave['r']) + len(specobj.wave['z'])
        qsospec = np.zeros(nwaves,dtype={'names':('wave', 'flux', 'ivar', 'model'), 'formats':('>f8', '>f8', '>f8', '>f8')})
        qsospec['wave'] = np.concatenate([specobj.wave['b'], specobj.wave['r'], specobj.wave['z']])
        qsospec['flux'] = np.concatenate([specobj.flux['b'][qindx], specobj.flux['r'][qindx], specobj.flux['z'][qindx]])
        qsospec['ivar'] = np.concatenate([specobj.ivar['b'][qindx], specobj.ivar['r'][qindx], specobj.ivar['z'][qindx]])
        ind = np.argsort(qsospec['wave'])
        qsospec['wave'] = qsospec['wave'][ind]
        qsospec['flux'] = qsospec['flux'][ind]
        qsospec['ivar'] = qsospec['ivar'][ind]

    # Get the balinfo for the figure 
    balinfo = baltable.cattobalinfo(balcat[bindx])
        
    pcaout = np.append(balcat['PCA_COEFFS'][bindx], np.asarray([balcat['PCA_CHI2'][bindx], -999.]))

    pcafit = fitbal.createpcatemplate(pcaeigen, pcaout)

    plotbal(qsospec, pcafit, pcaeigen, balinfo, zspec, pcaout, targetid=targetid, lam1=lam1, lam2=lam2)


def plotbal(qsospec, pcafit, pcaeigen, balinfo, zspec, pcaout, lam1=1340, lam2=1680, sdssname="", targetid="", savefig=False, outfig="balfit.png", verbose=False, plotvar=True):
    '''
    Plot a BAL spectrum, mark troughs, and overplot the PCA fit

    Parameters
    ----------
    qsospec : np.array()
        SDSS QSO spectrum
    pcafit : 1-D float array
        PCA template fit to QSO
    pcaeigen : np.array()
        PCA wavelength and components
    balinfo : dict
        dictionary with BAL parameters
    zspec : float
        redshift of QSO
    pcaout : 1-d float array
        PCA coefficients + PCA chi2 + SDSS chi2
    lam1 : float
        first wavelength to plot
    lam2 : float
        last wavelength to plot
    sdssname : string
        SDSS_NAME of QSO 
    targetid : int
        DESI TARGETID
    savefig : bool
        save the figure? 
    outfig : string
        name of the output figure 
    verbose : bool
        turn on or off some progress reporting
    plotvar : bool
        add variance to plot

    Returns
    -------
    none
    '''

    fig, ax = plt.subplots(1, figsize=(12,8))

    try:
        lam_z = qsospec['wave']/(1+zspec)
    except ValueError:
        try:
            lam_z = np.power(10, qsospec['loglam'])/(1+zspec)
        except:
            print("Error: 'wave' and 'loglam' not found")

    # use mm so plot autoscales to just the desired wavelength range
    mm = lam_z < lam2
    mm = mm * lam_z > lam1

    ax.plot(lam_z[mm], qsospec['flux'][mm], label="Data")
    ax.set_xlim(lam1, lam2)
    ax.plot(pcaeigen['WAVE'], pcafit, label="PCA Fit")
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if ymin < -1.:
        ymin = -1.
        ax.set_ylim(ymin, ymax) 
    if plotvar: 
        #ax.plot(np.power(10, qsospec['loglam'][mm])/(1+zspec),
        ax.plot(lam_z[mm], 
             np.sqrt(1./qsospec['ivar'][mm]), 'r-', label="stdev")

    # If pcaout[-1] > 0, then SDSS and there will be a model
    if pcaout[-1] > 0.: 
        ax.plot(lam_z, qsospec['model'], 'r:', label="Model") 

    # draw vertical lines at BAL limits
    ymax2 = 0.75*ymax
    drawtroughs(ax, balinfo, ymin, ymax2)

    ax.set_xlabel('Wavelength', fontsize=14)
    ax.set_ylabel('Flux', fontsize=14)

    #sdsschisq = fitbal.sdsschisq(qsospec, zspec)

    # add label
    if len(sdssname) > 0: 
        x1 = xmin + 0.05*(xmax-xmin)
        y1 = ymin + 0.95*(ymax-ymin)
        ax.text(x1,y1,sdssname, fontsize=14) 
    elif isinstance(targetid, (int, np.int32, np.int64)):
        x1 = xmin + 0.05*(xmax-xmin)
        y1 = ymin + 0.95*(ymax-ymin)
        tstring = "TARGETID {0}".format(targetid)
        ax.text(x1,y1,tstring, fontsize=14) 
    x2 = xmin + 0.05*(xmax-xmin)
    y2 = ymin + 0.9*(ymax-ymin)
    if pcaout[-1] > 0.: 
        zlab = r"z = {0:.2f}, $\chi_P^2$ = {1:.2f} $\chi_S^2$ = {2:.2f}".format(zspec, pcaout[-2], pcaout[-1])
    else: 
        zlab = r"z = {0:.2f}, $\chi_P^2$ = {1:.2f}".format(zspec, pcaout[-2]) 
    ax.text(x2,y2,zlab, fontsize=14) 
    x3 = xmin + 0.05*(xmax-xmin)
    y3 = ymin + 0.85*(ymax-ymin)
    ilab = "BI = {0:.0f} ({1:.0f}) AI = {2:.0f} ({3:.0f})".format(balinfo['BI_CIV'], balinfo['BI_CIV_ERR'], balinfo['AI_CIV'], balinfo['AI_CIV_ERR'])
    ax.text(x3,y3,ilab, fontsize=14) 

    # add legend
    ax.legend(loc='upper right', fontsize=14)
 
    if savefig: 
        fig.savefig(outfig) 
        if verbose: 
            print("Wrote output file {0}".format(outfig))

    plt.show()
    plt.close() 

def drawtroughs(ax, balinfo, ymin, ymax): 
    '''
    Add troughs from balinfo to a plot 

    Parameters
    ----------
    ax : AXES 
        current plot
    balinfo : dict
        dictionary with BAL parameters
    ymin : float
        minimum y value of line
    ymax : float
        maximum y value of line

    Returns
    -------
    none
    '''

    # draw trough boundaries
    for v in balinfo['VMIN_CIV_2000']:
        if v > 0.:
            w = bc.lambdaCIV*(1.-v/c)
            ax.plot(np.array([w, w]), np.array([ymin, ymax]), 'k--')
    for v in balinfo['VMAX_CIV_2000']:
        if v > 0.:
            w = bc.lambdaCIV*(1.-v/c)
            ax.plot(np.array([w, w]), np.array([ymin, ymax]), 'k--')
    for v in balinfo['VMIN_SIIV_2000']:
        if v > 0.:
            w = bc.lambdaSiIV*(1.-v/c)
            ax.plot(np.array([w, w]), np.array([ymin, ymax]), 'k--')
    for v in balinfo['VMAX_SIIV_2000']:
        if v > 0.:
            w = bc.lambdaSiIV*(1.-v/c)
            ax.plot(np.array([w, w]), np.array([ymin, ymax]), 'k--')

    for v in balinfo['VMIN_CIV_450']:
        if v > 0.:
            w = bc.lambdaCIV*(1.-v/c)
            ax.plot(np.array([w, w]), np.array([ymin, ymax]), 'k:')
    for v in balinfo['VMAX_CIV_450']:
        if v > 0.:
            w = bc.lambdaCIV*(1.-v/c)
            ax.plot(np.array([w, w]), np.array([ymin, ymax]), 'k:')
    for v in balinfo['VMIN_SIIV_450']:
        if v > 0.:
            w = bc.lambdaSiIV*(1.-v/c)
            ax.plot(np.array([w, w]), np.array([ymin, ymax]), 'k:')
    for v in balinfo['VMAX_SIIV_450']:
        if v > 0.:
            w = bc.lambdaSiIV*(1.-v/c)
            ax.plot(np.array([w, w]), np.array([ymin, ymax]), 'k:')

