"""
bal.fitdesipca
==============

Routines to read DESI spectrum file, fit PCA components to QSOs in DESI data, 
and calculate BAL properties

2019 Summer by Victoria Niu
DESI Calculate BAL Parameters Function adapted from calcbalparams.py 

Two main function to run balfinder: classifydesiqso and classifyoneqso

"""

import numpy as np
import fitsio
import fitbal
import balconfig as bc
import baltable
import desispec.io
from desispec.coaddition import coadd_cameras
from collections import defaultdict


def desibalfinder(specfilename, altbaldir=None, overwrite=True, verbose=False): 
    '''
    Find BALs in DESI quasars
    1. Identify all objects classified as quasars that are in the redshift
       range to identify BAL features 
    2. Create an output BAL catalog and populate it with basic QSO data
    3. Run the BAL finder on each quasar and add results to the table

    Parameters
    ----------
    specfilename : filename
        FITS file with DESI spectra
    altbaldir : string, optional
        alternate output directory for catalog (default is None)
    overwrite : bool, optional
        Overwrite the BAL catalog if it exists? (default is True)
    verbose : bool, optional
        Provide verbose output? (default is False)

    Returns
    -------
    none
    ''' 

    # Read in the DESI spectra 
    specobj = desispec.io.read_spectra(specfilename)

    # See if 3 cameras are coadded, and coadd them if they are not: 
    # (at least some of the mock data are not coadded) 
    if 'brz' not in specobj.wave.keys():
        specobj = coadd_cameras(specobj, cosmics_nsig=None)

    # Read in the redshift data 
    if 'spectra-' in specfilename: 
        specshort = specfilename[specfilename.rfind('spectra-'):specfilename.rfind('.fits')]
        zshort = specshort.replace('spectra', 'zbest')
        zfilename = specfilename.replace(specshort, zshort)
    elif 'coadd-' in specfilename: 
        specshort = specfilename[specfilename.rfind('coadd-'):specfilename.rfind('.fits')]
        zshort = specshort.replace('coadd', 'zbest')
        zfilename = specfilename.replace(specshort, zshort)
    else:
        print("Error: unable to find redshift file for {}".format(specfilename))
        return

    zs = fitsio.read(zfilename)

    # Identify the spectra classified as quasars based on zs and within 
    # the nominal redshift range for BALs
    # BAL_ZMIN = 1.57
    # BAL_ZMAX = 5.6
    zmask = zs['Z'] > bc.BAL_ZMIN
    zmask = zmask*(zs['Z'] < bc.BAL_ZMAX)
    zmask = zmask*(zs['SPECTYPE'] == b'QSO')

    zqsos = []
    dd = defaultdict(list)
    for index, item in enumerate(zs['TARGETID']):
        if zmask[index]: 
            dd[item].append(index)
            zqsos.append(index)

    fm = specobj.fibermap

    # Create a list of the indices in specobj on which to run balfinder
    qsos = [index for item in fm["TARGETID"] for index in dd[item] if item in dd]

    # Initialize the BAL table with all quasars in 'qsos' 
    if 'spectra-' in specfilename: 
        balshort = specshort.replace('spectra-', 'baltable-')
    elif 'coadd-' in specfilename: 
        balshort = specshort.replace('coadd-', 'baltable-')
    else:
        print("Error: unable to interpret {}".format(specfilename))
        return

    baltmp = specfilename.replace(specshort, balshort) 
    if altbaldir is not None: 
        balfilename = os.path.join(altbaldir + "/", baltmp[baltmp.rfind("baltable-")::])
    else: 
        balfilename = baltmp

    print("Output BAL catalog:", balfilename)

    baltable.initbaltab_desi(fm[qsos], zs[zqsos], balfilename, overwrite=overwrite)

    balhdu = fits.open(balfilename) 

    # Read in the eigenspectra 
    pcaeigen = fitsio.read(bc.pcaeigenfile)

    # Loop through the QSOs and run the BAL finder
    for i in range(len(qsos)): 
        qso = qsos[i]
        zspec = zs['Z'][dd[fm['TARGETID'][qso]]][0]
        qsospec = np.zeros(len(specobj.wave['brz']),dtype={'names':('wave', 'flux', 'ivar', 'model'), 'formats':('>f8', '>f8', '>f8', '>f8')})
        qsospec['wave'] = specobj.wave['brz']
        qsospec['flux'] = specobj.flux['brz'][qso]
        qsospec['ivar'] = specobj.ivar['brz'][qso]
        qsospec['model'] = np.zeros(len(specobj.wave['brz'])) # add to match SDSS format
        targetid = fm['TARGETID'][qso]
        info, pcaout, mask = fitbal.calcbalparams(qsospec, pcaeigen, zspec)
        # update baltable
        balhdu = baltable.updatebaltab_desi(targetid, balhdu, info, pcaout)
        if verbose: 
            print("{0} Processed {1} at z = {2:.2f}: AI_CIV = {3:.0f}, BI_CIV = {4:.0f}".format(i, targetid, zspec, info['AI_CIV'], info['BI_CIV']))

    if verbose: 
        print("Wrote BAL catalog {0}".format(balfilename))

    balhdu.writeto(balfilename, overwrite=True)




# ------------- everything below here is deprecated -------------

def classifydesiqso(spechdu, zhdu, pcahdu):
    '''
    Find all quasar spectra in the DESI spectrum file and classify them as 
    either BAL and non-BAL quasars. The output are two python dictionaries that 
    contains information about the BALs and NONBALs.
    
    Parameters
    ----------
    spechdu: FITS HDU returned by astropy
        HDU for all spectra of targets in one DESI file
    zhdu: FITS HDU returned by astropy
        HDU for all redshifts of targets in one DESI file
    pcahdu: FITS HDU returned by astropy
        HDU for PCA eigenvectors
    
    The result of the classification will be printed out in the end of the function
    
    Returns
    -------
    BAL: 2-D array of dictionaries
        each dictionary in the array contains information of one BAL quasar 
        and its BAL information
    NON_BAL: 2-D array of dictionaries
        each dictionary in the array contains information of one normal quasar 
        without BAL features
    Dictionaries contain: TargetID, redshift, signal-to-noise, quasar type, 
        chi_sq, wave, flux, pcafit, mask, ivar, calcinfo
    '''

    qsolist, z, targetnames, signal_noise = finddesiqso(spechdu, zhdu)
    bal = []
    non_bal =[]
    if len(qsolist)!=0:
        wave, flux, ivar = getqsospectra(spechdu, zhdu, qsolist)
        for i in range(len(flux)):
            diction = getdesifitpca(wave, flux[i], ivar[i], z[i], pcahdu)
            diction['TARGETID']=targetnames[i]
            diction['Z']=z[i]
            diction['SIGNAL_TO_NOISE']=signal_noise[i]
            if diction['TYPE']=='BAL':
                bal.append(diction)
            elif diction['TYPE']=='NON-BAL':
                non_bal.append(diction)
            else:
                print('Quasar type was not set!')
        
        BAL = np.array(bal)
        NON_BAL = np.array(non_bal)
        
        print('Result: In this spectrum file, there are ', len(qsolist), ' quasars. ', len(BAL), ' are BAL quasars with clear BAL feature around CIV emission line. ', len(NON_BAL), ' are normal quasars without BAL.')
    else:
        print('NO Quasar spectra in the spectrum file.')
        BAL= np.array([]) 
        NON_BAL= np.array([])
        
    return BAL, NON_BAL


def finddesiqso(spechdu, zhdu):
    '''
    A DESI spectrum file may have spectra that are not quasars. 
    Find all QSO spectra in one DESI file. 
    Input a spectrum file and the corresponding redshift catalog
    Returns the list of QSOs, their redshifts, TARGETIDs, and SNR of each
    
    Parameters
    ----------
    spechdu: FITS HDU
              HDU for spectra of DESI targets
    zhdu: FITS HDU
           HDU for redshifts of DESI targets
    
    Returns
    -------
    new_index: list
            index of targets whose desi_mask=='QSO' (in spectrum fibermap)
    z: 1-D float array
            list of the redshifts of each quasar
    targetnames: list
            QSO TARGETIDs 
    signal_noise: list
            average SNR of each spectrum (over entire wavelength range)
    '''

    newzqsolist = np.where((zhdu['FIBERMAP'].data['DESI_TARGET']==desi_mask['QSO']) & (zhdu['ZBEST'].data['Z']>1.5) & (zhdu['ZBEST'].data['Z']<5.0))[0]
    
    targetnames = zhdu['FIBERMAP'].data['TARGETID'][newzqsolist]
    z = zhdu['ZBEST'].data['Z'][newzqsolist]
    
    # get the spectrum IDs of these QSOs in spectrum list
    # if there are repeated spectra of the same QSO, pick the highest 
    # signal-to-noise one
    new_index = []
    signal_noise = []
    for i in range(len(targetnames)):
        repindx = np.where(spechdu['FIBERMAP'].data['TARGETID']==targetnames[i])[0]                   # indices of repeated spectrums
        maxsign, signnoise = maxsigntonoise(spechdu, repindx)
        new_index.append(maxsign)
        signal_noise.append(signnoise)
    
    # check length
    if len(new_index)!=len(z) or len(new_index)!=len(targetnames) or len(z)!=len(targetnames):
        print('ERROR! Length does not match!')

    return new_index, z, targetnames, signal_noise




def getqsospectra(spechdu, zhdu, speclist):
    '''
    Get spectra of all QSOs in one DESI spectrum file
    
    Parameters
    ----------
    spechdu: FITS HDU returned by astropy
         HDU for DESI spectra
    zhdu: FITS HDU returned by astropy
         HDU for the redshifts of DESI targets
    speclist: 1-D int array
         index of DESI targets that are QSOs with redshift>1.5 among all 
         targets in one spectrum file
    zlist: 1-D int array
         row indice of DESI targets that are QSOs with redshift>1.5 in the redshift file of the spectrums
    targetnames: 1-D int array
         list of these QSOs' TARGETID
    
    speclist, zlist, targetnames should point to the same QSO in row
    
    Returns
    -------
    wave: 1-D float array
         full wavelength of the spectra
    flux: 2-D float array
         full flux of all QSO spectras
    ivar: 2-D float array
         full inverse-of-variance of all QSO spectras
    z: 1-D float array
         redshifts of all QSOs
    '''

    # get lists of all QSO waves, fluxes and ivars
    bivar = spechdu['B_IVAR'].data[speclist]
    rivar = spechdu['R_IVAR'].data[speclist]
    zivar = spechdu['Z_IVAR'].data[speclist]
    
    bwave = spechdu['B_WAVELENGTH'].data
    rwave = spechdu['R_WAVELENGTH'].data
    zwave = spechdu['z_WAVELENGTH'].data
    
    bflux = spechdu['B_FLUX'].data[speclist]
    rflux = spechdu['R_FLUX'].data[speclist]
    zflux = spechdu['Z_FLUX'].data[speclist]
    
    fwave, midpoints = combinewaves(bwave, rwave, zwave)        # fwave is full wave
    
    # get for all wave
    flux = []
    ivar = []
    for i in range(len(speclist)):
        # first, add three spectrums into one
        fflux = combinefluxes(bflux[i], rflux[i], zflux[i], midpoints)
        flux.append(fflux)
        fivar = combineivars(bivar[i], rivar[i], zivar[i], midpoints)
        ivar.append(fivar)
        
    return fwave, flux, ivar






def desibalparams(loglam, flux, ivar, pcaeigen, zspec, maxiter=10, verbose=False):
    '''
    Adapted from calcbalparams(). 
    Iteratively fit BAL features in QSO spectrum at redshift zspec with 
       eigenvectors in pcaeigen and return BAL data
    Returns BAL info, PCA fit coefficients, and mask of BAL troughs
    
    Parameters
    ----------
    loglam: 1-D float array
        linear log(wavelength)
    flux: 1-D float array
        full flux of this QSO, nonlinear to match log(wavelength) array in column
    ivar: 1-D float array
        full ivar of this QSO, nonlinear to match log(wavelength) array in column
    pcaeigen: np.array()
        PCA wavelength and components
    zspec: float
        redshift of this QSO
        
    Returns
    -------
    calcinfo : array
        BAL properties
    calcpcaout : 1-D float array
        coefficients from PCA fit to QSO spectrum + PCA chisq + SDSS chisq
    calcmask : boolean array
        mask of BAL troughs
    '''
    wave_rest = np.power(10, loglam)/(1+zspec)
    # if wave_rest does not extend past CIV, return empty balinfo
    if wave_rest[-1] < bc.lambdaCIV:
        print("Spectrum does not extend red enough to include the CIV line")
        raise RuntimeError("Spectrum does not extend red enough to include the CIV line")
    if wave_rest[0] > bc.lambdaCIV:
        print("Spectrum does not extend blue enough to include the CIV line")
        raise RuntimeError("Spectrum does not extend blue enough to include the CIV line")

    # Interpolate the PCA components onto the rest wavelength values
    pca0 = np.interp(wave_rest, pcaeigen['WAVE'], pcaeigen['PCA0'])
    pca1 = np.interp(wave_rest, pcaeigen['WAVE'], pcaeigen['PCA1'])
    pca2 = np.interp(wave_rest, pcaeigen['WAVE'], pcaeigen['PCA2'])
    pca3 = np.interp(wave_rest, pcaeigen['WAVE'], pcaeigen['PCA3'])
    pca4 = np.interp(wave_rest, pcaeigen['WAVE'], pcaeigen['PCA4'])

    # index of shortest wavelength will be the wavelength of either
    # bc.BAL_LAMBDA_MIN or wave_rest[0]
    indx1 = 0
    if bc.BAL_LAMBDA_MIN > wave_rest[0]:
        indx1 = find_nearest(wave_rest, bc.BAL_LAMBDA_MIN) + 1
    # index of longest wavelength will be the wavelength of either
    # bc.BAL_LAMBDA_MAX or wave_rest[-1]
    indx2 = len(wave_rest) - 1
    if bc.BAL_LAMBDA_MAX < wave_rest[-1]:
        indx2 = find_nearest(wave_rest, bc.BAL_LAMBDA_MAX) - 1
    # 2-D array with interpolated PCA components
    ipca = np.array([pca0[indx1:indx2], pca1[indx1:indx2], pca2[indx1:indx2],
                     pca3[indx1:indx2], pca4[indx1:indx2]])

    # data over the same range
    balwave = wave_rest[indx1:indx2]
    balspec = flux[indx1:indx2]
    balivar = ivar[indx1:indx2]

    warnings.filterwarnings("ignore",category=RuntimeWarning) # eliminate divide by zero warning
    balerror = np.nan_to_num(np.sqrt(1/balivar))
    # balerror = 2.*balerror
    
    idata = np.array([balwave, balspec, balerror])
    calcmask = np.ones(indx2-indx1, dtype=bool)
    if verbose:
        print("calcbalparams: Min,max fit wavelengths: {0:.1f}, {1:.1f}".format(balwave[0], balwave[-1]))
    
    nmasked = sum(calcmask) + 1
    itr = 0
    while (itr < maxiter):
        # Fit for the PCA coefficients
        calcpcaout = fitpca(idata, zspec, ipca, calcmask)
        calcpcacoeffs = calcpcaout[:bc.NPCA]
        # calcmodel = createpcatemplate(pcaeigen, calcpcacoeffs)
        calcmodel = np.zeros(len(balwave), dtype=float)
        for i in range(len(calcpcacoeffs)):
            calcmodel += calcpcacoeffs[i]*ipca[i]
        if debug:
            print("calcbalparams: shapes ", ipca.shape, idata.shape, balwave.shape, balspec.shape, calcmodel.shape)
        # Solve for BAL parameters
        # calcinfo = calculatebalinfo(qsohdu, pcaeigen, zspec, calcpcacoeffs)
        calcinfo = calculatebalinfo(idata, calcmodel)
        # Create a mask based on those troughs
        calcmask = baltomask(calcinfo, balwave)
        if verbose:
            print(itr, nmasked)
        if sum(calcmask) == nmasked:
            itr = maxiter
        else:
            nmasked = sum(calcmask)
            itr = itr + 1
    return calcinfo, calcpcaout, calcmask




def getdesifitpca(wave, flux, ivar, zspec, pcaeigen):
    '''
    Run desibalparams() and get PCA fits(outspec) a single spectrum. 
    Return a dictionary with the BAL information

    Parameters
    ----------
    wave : 1-D float array
        full wavelength at all resolutions (linear)
    flux : 1-D float array
        full flux of one QSO spectrum
    ivar : 1-D float array
        full inverse-of-variance of one QSO spectrum
    zspec : float
        redshift of this quasar
    pcahdu : np.array
        PCA eigenvectors
    
    Returns
    -------
    diction: dictionary
        contain all BAL information of this quasar (quasar type, chi_sq, wave, flux, pcafit, mask, ivar, calcinfo)
    
    '''

    diction = {}
    
    loglam, nflux, nivar = nonlinearwave(wave, flux, ivar)
    calcinfo, calcpcaout, calcmask = desibalparams(loglam, nflux, nivar, pcaeigen, zspec, maxiter=10, verbose=False)

    wave_rest = np.power(10, loglam)/(1+zspec)
    pcawave = pcahdu[1].data['wave']
    outspec = createpcatemplate(pcahdu, calcpcaout)
    lim1 = np.searchsorted(wave_rest, 1340)
    lim2 = np.searchsorted(wave_rest, 1600)
    lim3 = np.searchsorted(pcawave, 1340)
    lim4 = np.searchsorted(pcawave, 1600)
    
    # for redshift around z=1.5
    if lim1 == 0:
        lim1 = np.searchsorted(wave_rest, 1440)
        lim3 = np.searchsorted(pcawave, 1440)
            
    balspec = outspec[lim3:lim4]
    balmask = calcmask[lim3:lim4]
            
    balflux = nflux[lim1:lim2]
    balivar = nivar[lim1:lim2]
    balwave = wave_rest[lim1:lim2]
    
    # standardize with length (some length may be 771 due to rounding)
    if len(balwave)>770:
        balwave = balwave[0:770]
    if len(balflux)>770:
        balflux = balflux[0:770]
    if len(balivar)>770:
        balivar = balivar[0:770]
    if len(balmask)>770:
        balmask = balmask[0:770]
    if len(balspec)>770:
        balmask = balmask[0:770]
    
    if False in balmask:
        diction['TYPE']='BAL'
    else:
        diction['TYPE']='NON-BAL'
    
    diction['CHI_SQ']=calcpcaout[-1]
    diction['WAVE']=balwave
    diction['FLUX']=balflux
    diction['PCAFIT']=balspec
    diction['MASK']=balmask
    diction['IVAR']=balivar
    diction['CALCINFO']=calcinfo
    
    return diction




def classifyoneqso(wave, flux, ivar, zspec, pcahdu):
    '''
    Classify the category (BAL or non-BAL) of one quasar with given spectrum and identify its BAL features
    
    Parameters
    ----------
    wave: 1-D float array or 3-D float array
        the wavelength of the quasar spectra. It can be an array with three wavelength(b,r,z) or an array of one full wavelength
    flux: 1-D float array or 3-D float array
        the flux of the quasar. It can be an array with three flux(b,r,z) or an array of one full flux
    ivar: 1-D float array or 3-D float array
        the inverse variance of the quasar. It can be an array with three ivar(b,r,z) or an array of one full ivar
    zspec: float
        redshift of the quasar
    pcahdu: FITS HDU returned by astropy
        HDU for PCA eigenvectors
        
    Returns
    -------
    diction: python dictionary
        contain BAL information of this quasar (quasar type, chi_sq, wave, flux, pcafit, mask, ivar, calcinfo, redshift)
        The targetID and signal-to-noise are not included since these two variables should have been identified before calling the function
    '''
    if len(flux)==3 and len(wave)==3 and len(ivar)==3:
        fwave, midpoints = combinewaves(wave[0], wave[1], wave[2])
        fflux = combinefluxes(flux[0], flux[1], flux[2], midpoints)
        fivar = combineivars(ivar[0], ivar[1], ivar[2], midpoints)
        diction = getdesifitpca(fwave, fflux, fivar, zspec, pcahdu)
    else:
        diction = getdesifitpca(wave, flux, ivar, zspec, pcahdu)
        
    diction['Z']=zspec
    if diction['TYPE']=='BAL':
        print('BAL Quasar')
    elif diction['TYPE']=='NON-BAL':
        print('NON-BAL Quasar')
    else:
        print('Quasar type was not set!')
    
    return diction

'''
Below are extra functions written for checking all spectrum files in one path

The results will be written into a fits file and stored in the outputs directory
'''



import os
import datetime
from glob import glob
from astropy.io import fits
from astropy.table import Table


def getdesispectra(spectra_dir, pcahdu):
    '''
    Get spectrum files of DESI data, check the files, and search for BAL QSOs in the spectra
    The result of found BAL info will be written into a fits file
    
    Parameters
    ----------
    spec_dir: string
        the path of DESI data
    pcahdu: FITS HDU returned by astropy
        HDU for PCA eigenvectors
    
    Returns
    -------
    none
    '''
    spec_dir = glob(spectra_dir+'/*/')
    for i in range(len(spec_dir)):
        spec_path = glob(spec_dir[i]+'/*')
        for k in range(len(spec_path)):
            specfile = glob(spec_path[k]+'/spectra*.fits')
            zfile = glob(spec_path[k]+'/zbest*.fits')
            if len(specfile)==0 or len(zfile)==0:
                if len(specfile)==0:
                    print(spec_path[k], ' has no spectrum file.')
                if len(zfile)==0:
                    print(spec_path[k], ' has no redshift file.')
            elif len(specfile)==len(zfile)==0:
                print(spec_path[k], ' has no spectrum and redshift files.')
            elif len(specfile)>1 or len(zfile)>1:
                print(spec_path[k], ' has more than one spectrum file.')
            else:
                filename = spec_path[k][-4:]
                if not os.path.isfile(filename+'.fits'):
                    print('path: ', spec_path[k])
                    zhdu = fits.open(zfile[0])
                    spechdu = fits.open(specfile[0])
                    BAL, NON_BAL= classifydesiqso(spechdu, zhdu, pcahdu)
                    if len(BAL)>0:
                        writetofits(spec_path[k], BAL)
                        

# define a function to write the BAL templates into a fits file
def writetofits(spec_path, BAL):
    '''
    Write the output BAL info from getdesispectra into a fits directory, stored in outputs directory
    
    Parameters
    ----------
    spec_path: string
        the path of the spectrum file
    BAL: 2-D array of python dictionaries
        output BAL info from classifydesiqso
        
    Returns
    -------
    none 
    '''
    filename = spec_path[-4:]+'.fits'
    output_dir = 'outputs/'
    
    # write values in dictionary into a column
    targetid = []
    chi_sq = []
    signoise = []
    z = []
    flux = []
    ivar = []
    outspec = []
    for i in range(len(BAL)):
        targetid.append(BAL[i]['TARGETID'])
        chi_sq.append(BAL[i]['CHI_SQ'])
        signoise.append(BAL[i]['SIGNAL_TO_NOISE'])
        z.append(BAL[i]['Z'])
        flux.append(BAL[i]['FLUX'])
        ivar.append(BAL[i]['IVAR'])
        outspec.append(BAL[i]['PCAFIT'])
    
    # create columns
    col1 = fits.Column(name = 'TARGETID', format = 'K', array=targetid)
    col2 = fits.Column(name = 'CHI_SQ', format = 'D', array=chi_sq)
    col3 = fits.Column(name = 'SIGNAL_TO_NOISE', format = 'D', array=signoise)
    col4 = fits.Column(name = 'Z', format = 'D', array=z)
    col5 = fits.Column(name = 'FLUX', format = str(len(flux[0]))+'D', array=flux)
    col6 = fits.Column(name = 'IVAR', format = str(len(ivar[0]))+'D', array=ivar)
    col7 = fits.Column(name = 'OUTSPEC', format = str(len(outspec[0]))+'D', array=outspec)
    
    
    # create a HDU Table
    
    tabhdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5, col6, col7])
    
    tabhdu.header['CDELT1'] = 0.3
    tabhdu.header['CRVAL1'] = BAL[0]['WAVE'][0]
    tabhdu.header['EXTNAME'] = 'BAL-output.fits'

    tabhdu.header.add_comment('Outputs of BALs from Balfinder.')
    tabhdu.header.add_history('Created on '+datetime.datetime.now().strftime("%Y-%m-%d"))
    
    tabhdu.writeto(output_dir+filename, overwrite=True)

def maxsigntonoise(spechdu, rep_indice):
    '''
    DESI may observe the same QSO multiple times. 
    Identify which spectrum is the highest SNR
    
    Parameters
    ----------
    spechdu: FITS HDU returned by astropy
             HDU for QSO spectrum
    rep_indice: 1-D int array
             row indice of repeated spectra of one quasar (index in the fibermap)
    
    Returns
    -------
    rep_indice[maxsign]: int
             index of the spectrum with the highest signal-to-noise (the spectrum will be used to make PCA fit)
    avgsign[maxsign]: float
             value of the maximum signal-to-noise of this quasar
    '''

    avgsign = np.zeros(len(rep_indice))

    for i in range(len(rep_indice)):
        bavg=np.average(spechdu['B_IVAR'].data[rep_indice[i]]*spechdu['B_FLUX'].data[rep_indice[i]])
        ravg=np.average(spechdu['R_IVAR'].data[rep_indice[i]]*spechdu['R_FLUX'].data[rep_indice[i]])
        zavg=np.average(spechdu['Z_IVAR'].data[rep_indice[i]]*spechdu['Z_FLUX'].data[rep_indice[i]])
        avgsign[i]=np.average([bavg, ravg, zavg])

    maxsign = np.argmax(avgsign)
    return(rep_indice[maxsign], avgsign[maxsign])


def combinewaves(bwave, rwave, zwave):
    '''
    Combine the three wavelength arrays into one
    
    Parameters
    ----------
    bwave: 1-D float array
           wavelength range at B resolution
    rwave: 1-D float array
           wavelength range at R resolution
    zwave: 1-D float array
           wavelength range at Z resolution
    
    three variables above can be directly read from spechdu
    
    Returns
    -------
    full_wave: 1-D float array
           full wavelength range at all resolutions
    midpoints: 1-D int array, length=4
            midpoints of the overlap regions between bwave and rwave, and between rwave and zwave
    '''
    # between b and r wave
    br_start = np.searchsorted(bwave, rwave[0])                       # starting point of two spectrums overlap
    br_end = np.searchsorted(rwave, bwave[-1])                        # ending point of two spectrums overlap
    # midpoint
    br_midpoint = int((br_start+len(bwave)-1)/2)
    rb_midpoint = np.searchsorted(rwave, bwave[br_midpoint])
    bpart_wave = bwave[:br_midpoint]
    
    # between r and z wave
    rz_start = np.searchsorted(rwave, zwave[0])                       # starting point of two spectrums overlap
    rz_end = np.searchsorted(zwave, rwave[-1])                        # ending point of two spectrums overlap
    rz_midpoint = int((rz_start+len(rwave)-1)/2)
    zr_midpoint = np.searchsorted(zwave, rwave[rz_midpoint])
    rpart_wave = rwave[rb_midpoint:rz_midpoint]
    zpart_wave = zwave[zr_midpoint:]
    
    # add all wave together
    rb_wave = np.append(bpart_wave, rpart_wave)
    full_wave = np.append(rb_wave, zpart_wave)
    
    midpoints = np.array([br_midpoint, rb_midpoint, rz_midpoint, zr_midpoint])
    # return the full wave and intercep midpoints (put to an array)
   
    return full_wave, midpoints




def combinefluxes(bflux, rflux, zflux, midpoints):
    '''
    Combine the three flux arrays into one
    
    Parameters
    ----------
    bflux: 1-D float array
        flux at B wavelength range
    rflux: 1-D float array
        flux at R wavelength range
    zflux: 1-D float array
        flux at Z wavelength range
    
    three variables above can be directly read from spechdu
    
    midpoints: 1-D int array, length=4
        midpoints of the overlap regions between bwave and rwave, and between rwave and zwave
        
    
    Returns
    -------
    fflux: 1-D float array
        full flux at all wavelength range
    '''
    # use the midpoints
    bpart = bflux[:midpoints[0]]
    rpart = rflux[midpoints[1]:midpoints[2]]
    zpart = zflux[midpoints[3]:]
    
    rb_flux = np.append(bpart, rpart)
    # full flux of this spectrum
    fflux = np.append(rb_flux, zpart)
    return fflux




def combineivars(bivar, rivar, zivar, midpoints):
    '''
    Combine the three inverse variance arrays into one
    
    Parameters
    ----------
    bivar: 1-D float array
        inverse-of-ivariance at B wavelength range
    rivar: 1-D float array
        inverse-of-ivariance at R wavelength range
    zivar: 1-D float array
        inverse-of-ivariance at Z wavelength range
    
    three variables above can be directly read from spechdu
    
    midpoints: 1-D int array, length=4
        midpoints of the overlap regions between bwave and rwave, and between rwave and zwave
    
    Returns
    -------
    fivar: 1-D float array
        full inverse-of-ivariance at all wavelength range
    '''
    # use the midpoints
    bpart = bivar[:midpoints[0]]
    rpart = rivar[midpoints[1]:midpoints[2]]
    zpart = zivar[midpoints[3]:]
    
    rb_ivar = np.append(bpart, rpart)
    # full ivar of this spectrum
    fivar = np.append(rb_ivar, zpart)
    return fivar


def nonlinearwave(wave, flux, ivar):
    '''
    DESI uses linear wavelength; SDSS use linear log(wavelength). 
    Convert from linear wavelength to linear in the log of the wavelength
    
    Parameters
    ----------
    wave: 1-D float array
        full wavelength at all resolutions (linear)
    flux: 1-D float array
        full flux at all resolutions (linear to match wavelength array in column)
    ivar: 1-D float array
        full inverse of variance at all resolutions(linear to match wavelength array in column)
    
    Returns
    -------
    wave_lam: 1-D float array
        linear log(wavelength)
    nonlinear_flux: 1-D float array
        flux changed to nonlinear to match log(wavelength) array in column
    nonlinear_ivar: 1-D float array
        ivar changed to nonlinear to match log(wavelength) array in column
    '''
    log10lam = np.log10(wave)
    wave_lam = np.arange(log10lam[0], log10lam[-1], 0.0001)
    wave_back = np.power(10, wave_lam)
    nonlinear_flux = np.interp(wave_back, wave, flux)
    nonlinear_ivar = np.interp(wave_back, wave, ivar)
    return wave_lam, nonlinear_flux, nonlinear_ivar


