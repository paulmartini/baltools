"""

baltools.createbaltruth
=======================

Create truth catalogs for DESI mocks with BAL features

TODO: Reconcile/merge with baltable? 

"""

from __future__ import print_function, absolute_import, division

from astropy.io import fits
import numpy as np
import os
import glob

NBI = 5
NAI = 17

# Mock data with BALs 

def createbaltruth(baldir): 

    # Open Truth Catalog
    truthfile = glob.glob(baldir+'truth*')[0]
    thdu = fits.open(truthfile) 

    # Open the BAL Template File
    baltempfile = '/global/cfs/projectdirs/desi/spectro/templates/basis_templates/v3.2/bal_templates_v3.0.fits'
    baltemphdu = fits.open(baltempfile)

    # Create an empty BAL Truth Catalog in the same directory as the Truth Catalog
    baltruthfile = truthfile.replace('truth', 'baltruth') 
    initbaltruth(thdu, baltruthfile) 
    print("Initialized truth catalog {0}".format(baltruthfile))
    bthdu = fits.open(baltruthfile) 

    # Identify if each TARGETID is a BAL, and if so get the TEMPLATEID
    baltemplateid = -1*np.ones(len(thdu['TRUTH'].data['TARGETID']), dtype=int)
    baltargetid = -1*np.ones(len(thdu['TRUTH'].data['TARGETID']), dtype=int)
    for i in range(len(thdu['TRUTH'].data)): # loop through TRUTH
        tmpindx = np.where(thdu['TRUTH'].data['TARGETID'][i] == thdu['BAL_META'].data['TARGETID'])[0]
        if len(tmpindx) == 1: 
            # If it is a BAL, get baltemplateid and update the BAL Truth Catalog
            baltemplateid[i] = thdu['BAL_META'].data['TEMPLATEID'][tmpindx[0]]
            baltargetid[i] = thdu['BAL_META'].data['TARGETID'][tmpindx[0]]
            balinfo = getbalinfo(baltemphdu['METADATA'].data[baltemplateid[i]]) 
            balinfo['BALTEMPLATEID'] = baltemplateid[i]
            bthdu = updatebaltable(i, bthdu, balinfo)

    # Write out the BAL Truth Catalog
    bthdu.writeto(baltruthfile, overwrite=True)
    print("Wrote outputfile {0}".format(baltruthfile))

def initbaltruth(truehdu, outputfile): 
    '''
    Create an empty BAL Truth Catalog table from a Truth Catalog

    Parameters
    ----------
    truehdu : FITS HDU
        HDU of fits truth catalog
    filename : string
        name of BAL truth catalog to create

    Returns
    -------
    none
    '''

    NROWS = len(truehdu[1].data)
    # Columns to copy from the Truth Catalog
    col0 = truehdu[1].columns['TARGETID']
    col1 = truehdu[1].columns['Z']

    # Collection of empty arrays to set up rest of the BAL catalog
    # PM Note: Might change to -1. to indicate empty field
    zfloat_col = np.zeros([NROWS], dtype=float)
    zint_col = np.zeros([NROWS], dtype=float)
    zfloat_bicol = np.zeros([NROWS, NBI], dtype=float)
    zfloat_aicol = np.zeros([NROWS, NAI], dtype=float)


    # 0. or 1. depending on TRUTH Catalog
    col25 = fits.Column(name='BALPROB', format='E', array=zfloat_col)
    col2 = fits.Column(name='BALTEMPLATEID', format='E', array=zfloat_col)

    # These quantities are from trough fit to the template
    col26 = fits.Column(name='BI_CIV', format='E', array=zfloat_col)
    col27 = fits.Column(name='ERR_BI_CIV', format='E', array=zfloat_col)
    col28 = fits.Column(name='NCIV_2000', format='J', array=zint_col)
    col29 = fits.Column(name='VMIN_CIV_2000', format='5E', array=zfloat_bicol)
    col30 = fits.Column(name='VMAX_CIV_2000', format='5E', array=zfloat_bicol)
    col31 = fits.Column(name='POSMIN_CIV_2000', format='5E', array=zfloat_bicol)
    col32 = fits.Column(name='FMIN_CIV_2000', format='5E', array=zfloat_bicol)

    col33 = fits.Column(name='AI_CIV', format='E', array=zfloat_col)
    col34 = fits.Column(name='ERR_AI_CIV', format='E', array=zfloat_col)
    col35 = fits.Column(name='NCIV_450', format='J', array=zint_col)
    col36 = fits.Column(name='VMIN_CIV_450', format='17E', array=zfloat_aicol)
    col37 = fits.Column(name='VMAX_CIV_450', format='17E', array=zfloat_aicol)
    col38 = fits.Column(name='POSMIN_CIV_450', format='17E', array=zfloat_aicol)
    col39 = fits.Column(name='FMIN_CIV_450', format='17E', array=zfloat_aicol)

    col40 = fits.Column(name='BI_SIIV', format='E', array=zfloat_col)
    col41 = fits.Column(name='ERR_BI_SIIV', format='E', array=zfloat_col)
    col42 = fits.Column(name='NSIIV_2000', format='J', array=zint_col)
    col43 = fits.Column(name='VMIN_SIIV_2000', format='5E', array=zfloat_bicol)
    col44 = fits.Column(name='VMAX_SIIV_2000', format='5E', array=zfloat_bicol)
    col45 = fits.Column(name='POSMIN_SIIV_2000', format='5E', array=zfloat_bicol)
    col46 = fits.Column(name='FMIN_SIIV_2000', format='5E', array=zfloat_bicol)

    col47 = fits.Column(name='AI_SIIV', format='E', array=zfloat_col)
    col48 = fits.Column(name='ERR_AI_SIIV', format='E', array=zfloat_col)
    col49 = fits.Column(name='NSIIV_450', format='J', array=zint_col)
    col50 = fits.Column(name='VMIN_SIIV_450', format='17E', array=zfloat_aicol)
    col51 = fits.Column(name='VMAX_SIIV_450', format='17E', array=zfloat_aicol)
    col52 = fits.Column(name='POSMIN_SIIV_450', format='17E', array=zfloat_aicol)
    col53 = fits.Column(name='FMIN_SIIV_450', format='17E', array=zfloat_aicol)


    tabhdu = fits.BinTableHDU.from_columns([col0, col1, col25, col2, col26,
                                        col27, col28, col29, col30, col31,
                                        col32, col33, col34, col35, col36,
                                        col37, col38, col39, col40, col41,
                                        col42, col43, col44, col45, col46,
                                        col47, col48, col49, col50, col51,
                                        col52, col53])

    # print(tabhdu.info())
    # print("BAL table has {0} rows".format(len(tabhdu[1].data))

    tabhdu.writeto(outputfile, overwrite=True) 


def updatebaltable(qindx, balhdu, info):
    '''
    Add balinfo to entry qindx in truth table balhdu

    Parameters
    ----------
    qindx : int
        index in balhdu
    balhdu : FITS hdu
        HDU for BAL truth table
    info : record array
        dictionary with BAL data from template file for this BAL

    Returns
    -------
    balhdu : FITS HDU
        fits hdu for updated bal catalog
    '''

    balhdu[1].data[qindx]['BALPROB'] = 1.
    balhdu[1].data[qindx]['BALTEMPLATEID'] = info['BALTEMPLATEID']

    balhdu[1].data[qindx]['BI_CIV'] = info['BI_CIV']
    balhdu[1].data[qindx]['ERR_BI_CIV'] = info['BI_CIV_ERR']
    balhdu[1].data[qindx]['NCIV_2000'] = info['NCIV_2000']
    balhdu[1].data[qindx]['VMIN_CIV_2000'] = info['VMIN_CIV_2000']
    balhdu[1].data[qindx]['VMAX_CIV_2000'] = info['VMAX_CIV_2000']
    balhdu[1].data[qindx]['POSMIN_CIV_2000'] = info['POSMIN_CIV_2000']
    balhdu[1].data[qindx]['FMIN_CIV_2000'] = info['FMIN_CIV_2000']

    balhdu[1].data[qindx]['AI_CIV'] = info['AI_CIV']
    balhdu[1].data[qindx]['ERR_AI_CIV'] = info['AI_CIV_ERR']
    balhdu[1].data[qindx]['NCIV_450'] = info['NCIV_450']
    balhdu[1].data[qindx]['VMIN_CIV_450'] = info['VMIN_CIV_450'][:NAI]
    balhdu[1].data[qindx]['VMAX_CIV_450'] = info['VMAX_CIV_450'][:NAI]
    balhdu[1].data[qindx]['POSMIN_CIV_450'] = info['POSMIN_CIV_450'][:NAI]
    balhdu[1].data[qindx]['FMIN_CIV_450'] = info['FMIN_CIV_450'][:NAI]

  #  balhdu[1].data[qindx]['BI_SIIV'] = info['BI_SIIV']
  #  balhdu[1].data[qindx]['ERR_BI_SIIV'] = info['BI_SIIV_ERR']
  #  balhdu[1].data[qindx]['NSIIV_2000'] = info['NSIIV_2000']
  #  balhdu[1].data[qindx]['VMIN_SIIV_2000'] = info['VMIN_SIIV_2000']
  #  balhdu[1].data[qindx]['VMAX_SIIV_2000'] = info['VMAX_SIIV_2000']
  #  balhdu[1].data[qindx]['POSMIN_SIIV_2000'] = info['POSMIN_SIIV_2000']
  #  balhdu[1].data[qindx]['FMIN_SIIV_2000'] = info['FMIN_SIIV_2000']

  #  balhdu[1].data[qindx]['AI_SIIV'] = info['AI_SIIV']
  #  balhdu[1].data[qindx]['ERR_AI_SIIV'] = info['AI_SIIV_ERR']
  #  balhdu[1].data[qindx]['NSIIV_450'] = info['NSIIV_450']
  #  balhdu[1].data[qindx]['VMIN_SIIV_450'] = info['VMIN_SIIV_450']
  #  balhdu[1].data[qindx]['VMAX_SIIV_450'] = info['VMAX_SIIV_450']
  #  balhdu[1].data[qindx]['POSMIN_SIIV_450'] = info['POSMIN_SIIV_450']
  #  balhdu[1].data[qindx]['FMIN_SIIV_450'] = info['FMIN_SIIV_450']

    return balhdu


def getbalinfo(array):
    '''
    Populate the balinfo dictionary from a row in a BAL catalog

    Parameters
    ----------
    array : data
        row of the BAL template file 

    Returns
    -------
    info : dict
        dictionary of BAL parameters
    '''

    info = {}

    info['BI_CIV'] = array['BI_CIV'] 
    info['BI_CIV_ERR'] = array['ERR_BI_CIV']
    info['NCIV_2000'] = array['NCIV_2000']
    info['VMIN_CIV_2000'] = array['VMIN_CIV_2000']
    info['VMAX_CIV_2000'] = array['VMAX_CIV_2000']
    info['POSMIN_CIV_2000'] = array['POSMIN_CIV_2000']
    info['FMIN_CIV_2000'] = array['FMIN_CIV_2000']
    info['AI_CIV'] = array['AI_CIV']
    info['AI_CIV_ERR'] = array['ERR_AI_CIV']
    info['NCIV_450'] = array['NCIV_450']
    info['VMIN_CIV_450'] = array['VMIN_CIV_450']
    info['VMAX_CIV_450'] = array['VMAX_CIV_450']
    info['POSMIN_CIV_450'] = array['POSMIN_CIV_450']
    info['FMIN_CIV_450'] = array['FMIN_CIV_450']
  #  info['BI_SIIV'] = array['BI_SIIV']
  #  info['BI_SIIV_ERR'] = array['ERR_BI_SIIV']
  #  info['NSIIV_2000'] = array['NSIIV_2000']
  #  info['VMIN_SIIV_2000'] = array['VMIN_SIIV_2000']
  #  info['VMAX_SIIV_2000'] = array['VMAX_SIIV_2000']
  #  info['POSMIN_SIIV_2000'] = array['POSMIN_SIIV_2000']
  #  info['FMIN_SIIV_2000'] = array['FMIN_SIIV_2000']
  #  info['AI_SIIV'] = array['AI_SIIV']
  #  info['AI_SIIV_ERR'] = array['ERR_AI_SIIV']
  #  info['NSIIV_450'] = array['NSIIV_450']
  #  info['VMIN_SIIV_450'] = array['VMIN_SIIV_450']
  #  info['VMAX_SIIV_450'] = array['VMAX_SIIV_450']
  #  info['POSMIN_SIIV_450'] = array['POSMIN_SIIV_450']
  #  info['FMIN_SIIV_450'] = array['FMIN_SIIV_450']

    return info
