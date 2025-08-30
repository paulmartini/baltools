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
from time import gmtime, strftime

from astropy.io import fits
import desispec.io
from desispec.coaddition import coadd_cameras,resample_spectra_lin_or_log
from collections import defaultdict

from baltools import fitbal
from baltools import balconfig as bc
from baltools import baltable
import getpass

def desibalfinder(specfilename, alttemp=False, altbaldir=None, altzdir=None, zfileroot='zbest', overwrite=True, verbose=False, release=None, usetid=True, format='healpix'): 
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
    alttemp : bool, optional
        Use Allyson Brozeller's optional alternate templates
    altbaldir : string, optional
        alternate output directory for catalog (default is None)
    altzdir : string, optional
        alternate directory to find redshift catalog (default in None, which will look in the coadd directory) 
    zfileroot : string, optional 
        root name for redshift catalog 
    overwrite : bool, optional
        Overwrite the BAL catalog if it exists? (default is True)
    verbose : bool, optional
        Provide verbose output? (default is False)
    release : string, optional
        Specifies the data release (default if None)
    usetid : bool, optional
        Only read specified TARGETIDs (default is True) 
    format : string, optional 
        Specifies either healpix or tile based (default is healpix)

    Returns
    -------
    none
    ''' 

    if release is None:
        release = 'kibo'
   
    if zfileroot is None: 
        if release == 'everest': 
            zfileroot = 'redrock' 
        elif release == 'himalayas': 
            zfileroot = 'zafter'
        else: 
            zfileroot = 'zbest' 

    # Define some variable names based on the type of input file
    if 'spectra-' in specfilename:
        specshort = specfilename[specfilename.rfind('spectra-'):specfilename.rfind('.fits')]
        zshort = specshort.replace('spectra', zfileroot)
        tshort = specshort.replace('spectra', 'truth')  # need this for mocks without resolution matrix 
    elif 'coadd-' in specfilename:
        specshort = specfilename[specfilename.rfind('coadd-'):specfilename.rfind('.fits')]
        zshort = specshort.replace('coadd', zfileroot)
    else:
        print("Error in desibalfinder(): unable to parse {}".format(specfilename))
        return

    zfilename = specfilename.replace(specshort, zshort)

    if altzdir is not None: 
        zfilename = os.path.join(altzdir, zshort+".fits") 

    # Define output file name and check if it exists if overwrite=False
    if 'spectra-' in specfilename:
        balshort = specshort.replace('spectra-', 'baltable-')
    elif 'coadd-' in specfilename:
        balshort = specshort.replace('coadd-', 'baltable-')
    else:
        print("Error in desibalfinder(): unable to interpret {}".format(specfilename))
        return

    baltmp = specfilename.replace(specshort, balshort)

    if altbaldir is not None:
        balfilename = os.path.join(altbaldir + "/", baltmp[baltmp.rfind("baltable-")::])
    else:
        balfilename = baltmp

    print("Output BAL catalog:", balfilename)

    if os.path.isfile(balfilename) and not overwrite:
        print("desibal(): Warning {0} exists and overwrite = False, so not re-running balfinder".format(balfilename))
        return

    # Get the QSO TARGETIDs
    zs = fitsio.read(zfilename)

    if verbose: 
        print("Read file {}".format(zfilename))

    tids = zs['TARGETID']
    if usetid: 
        specobj = desispec.io.read_spectra(specfilename, targetids=tids, skip_hdus=['B_RESOLUTION', 'R_RESOLUTION', 'Z_RESOLUTION'])
    else: 
        specobj = desispec.io.read_spectra(specfilename) 

    # See if 3 cameras are coadded, and coadd them if they are not:
    # (at least some of the mock data are not coadded)
    if 'brz' not in specobj.wave.keys():
        try:
            specobj = coadd_cameras(specobj) 
        except:
            if specobj.resolution_data is not None: 
                wave_min = np.min(specobj.wave['b'])
                wave_max = np.max(specobj.wave['z'])
                specobj = resample_spectra_lin_or_log(specobj,linear_step=0.8, wave_min =wave_min, wave_max =wave_max, fast = True)
                specobj = coadd_cameras(specobj)
                print("coadded_cameras using lispace resample")
            else:
                truthfile = specfilename.replace(specshort, tshort)
                if not os.path.isfile(truthfile):
                    print(f"Error: {truthfile} not found and resoution data not in {specfilename}")
                thdu = fits.open(truthfile)
                bdim = thdu['B_RESOLUTION'].data.shape[0]
                rdim = thdu['R_RESOLUTION'].data.shape[0]
                zdim = thdu['Z_RESOLUTION'].data.shape[0]
                bresdata = np.empty([specobj.flux['b'].shape[0], bdim, specobj.flux['b'].shape[1]], dtype=float)
                for i in range(specobj.flux['b'].shape[0]):
                    bresdata[i] = thdu['B_RESOLUTION'].data
                rresdata = np.empty([specobj.flux['r'].shape[0], rdim, specobj.flux['r'].shape[1]], dtype=float)
                for i in range(specobj.flux['r'].shape[0]):
                    rresdata[i] = thdu['R_RESOLUTION'].data
                zresdata = np.empty([specobj.flux['z'].shape[0], zdim, specobj.flux['z'].shape[1]], dtype=float)
                for i in range(specobj.flux['z'].shape[0]):
                    zresdata[i] = thdu['Z_RESOLUTION'].data
                specobj.resolution_data = {}
                specobj.resolution_data['b'] = bresdata
                specobj.resolution_data['r'] = rresdata
                specobj.resolution_data['z'] = zresdata
                #wave_min = np.min(specobj.wave['b'])
                wave_min = 3600.
                wave_max = np.max(specobj.wave['z'])
                specobj = resample_spectra_lin_or_log(specobj,linear_step=0.8, wave_min =wave_min, wave_max =wave_max, fast = True)
            
    # Identify the spectra classified as quasars based on zs and within 
    # the nominal redshift range for BALs
    # BAL_ZMIN = 1.57
    # BAL_ZMAX = 5.0
    zmask = zs['Z'] >= bc.BAL_ZMIN
    zmask = zmask*(zs['Z'] <= bc.BAL_ZMAX)
    
    if 'QSO' in np.unique(zs['SPECTYPE']) :
        zmask = zmask*(zs['SPECTYPE'] == 'QSO')
    else :
        zmask = zmask*(zs['SPECTYPE'] == 'bQSO')

    zqsos = []
    dd = defaultdict(list)
    # Create a dictionary with the index of each TARGETID in zs in 
    # the BAL redshift range. 
    # Create an array of indices in zs of those quasars 
    for index, item in enumerate(zs['TARGETID']):
        if zmask[index]: 
            dd[item].append(index)
            zqsos.append(index)

    if verbose: 
      print("Found {} quasars".format(np.sum(zmask)))

    fm = specobj.fibermap
    # Create a list of the indices in specobj on which to run balfinder
    # Note afterburners catalogs have different lengths from redrock catalogs
    if altzdir is None: 
        qsos = [index for item in fm["TARGETID"] for index in dd[item] if item in dd]
    else: 
        qsos = []
        for zindx in zqsos:
            tid = zs['TARGETID'][zindx]
            qindx = np.where(tid == fm['TARGETID'])[0][0]
            qsos.append(qindx)
            
    # Read in the eigenspectra 
    ### This has been changed to allow new components
    if alttemp:
        pcaeigen = fitsio.read(os.environ['HOME'] + '/Catalogs/PCA_Eigenvectors_Brodzeller.fits')
    else:
        pcaeigen = fitsio.read(bc.pcaeigenfile)

    # Initialize the BAL table with all quasars in 'qsos'
    baltable.initbaltab_desi(fm[qsos], zs[zqsos], balfilename, pcaeigen, overwrite=overwrite,release=release)

    balhdu = fits.open(balfilename) 

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
        info, pcaout, mask = fitbal.calcbalparams(qsospec, pcaeigen, zspec, verbose=False)
        # update baltable
        ### Code below has been changed to allow new argument
        balhdu = baltable.updatebaltab_desi(targetid, balhdu, info, pcaout, pcaeigen)
        if verbose: 
            # print("{0} Processed {1} at z = {2:.2f}: AI_CIV = {3:.0f}, BI_CIV = {4:.0f}".format(i, targetid, zspec, info['AI_CIV'], info['BI_CIV']))
            print("{0} Processed {1} at z = {2:.2f}: AI_CIV = {3:.0f}, BI_CIV = {4:.0f}, SNR_CIV = {5:.1f}".format(i, targetid, zspec, info['AI_CIV'], info['BI_CIV'], info['SNR_CIV']))

    balhdu.writeto(balfilename, overwrite=True)

    if verbose: 
        print("Wrote BAL catalog {0}".format(balfilename))

    lastupdate = "Last updated {0} UT by {1}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), getpass.getuser())
    fits.setval(balfilename, 'HISTORY', value=lastupdate, ext=1)
    balhdu.close() 
    del balhdu
    del qsospec
    del specobj

    if alttemp:
        balcatname = os.environ['HOME'] + '/Catalogs/PCA_Eigenvectors_Brodzeller.fits'
    else:
        balcatname = bc.pcaeigenfile
    fits.setval(balfilename, 'QSOTEMPS', value=balcatname, ext=1)

   
    
