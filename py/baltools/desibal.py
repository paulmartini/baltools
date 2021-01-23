"""

baltools.desibal
================

DESI balfinder code. Reads in a spectrum file, fits PCA components, 
identifies BAL troughs, measures their properties, and writes a 
BAL catalog. 

Adapted from classifydesiqso by Victoria Niu

"""

import os
import numpy as np
import fitsio
from astropy.io import fits
import desispec.io
from desispec.coaddition import coadd_cameras,resample_spectra_lin_or_log
from collections import defaultdict

from baltools import fitbal
from baltools import balconfig as bc
from baltools import baltable


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
    # Define some variable names based on the type of input file
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

    # Define output file name and check if it exists if overwrite=False
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

    if os.path.isfile(balfilename) and not overwrite:
        print("Bal catalog already exist")
        return

    # Read in the DESI spectra
    specobj = desispec.io.read_spectra(specfilename)
    # See if 3 cameras are coadded, and coadd them if they are not:
    # (at least some of the mock data are not coadded)
    if 'brz' not in specobj.wave.keys():
        try:
            specobj = coadd_cameras(specobj, cosmics_nsig=None)
        except:
            wave_min = np.min(specobj.wave['b'])
            wave_max = np.max(specobj.wave['z'])
            specobj = resample_spectra_lin_or_log(specobj,linear_step=0.8, wave_min =wave_min, wave_max =wave_max, fast = True)
            specobj = coadd_cameras(specobj, cosmics_nsig=None)
            print("coadded_cameras using lispace resample")
            


    zs = fitsio.read(zfilename)

    # Identify the spectra classified as quasars based on zs and within 
    # the nominal redshift range for BALs
    # BAL_ZMIN = 1.57
    # BAL_ZMAX = 5.6
    zmask = zs['Z'] > bc.BAL_ZMIN
    zmask = zmask*(zs['Z'] < bc.BAL_ZMAX)
    
    if 'QSO' in np.unique(zs['SPECTYPE']) :
        zmask = zmask*(zs['SPECTYPE'] == 'QSO')
    else :
        zmask = zmask*(zs['SPECTYPE'] == 'bQSO')

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

