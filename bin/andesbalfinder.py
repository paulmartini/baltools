#!/usr/bin/env python
  
'''
Create a BAL catalog based on a QSO catalog for the Andes release
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack
from glob import glob

import argparse
from time import gmtime, strftime

import fitsio
import desispec.io
from desispec.coaddition import coadd_cameras

import baltools
from baltools import balconfig as bc
from baltools import plotter, fitbal, baltable
from baltools import desibal as db

debug = True

os.environ['DESI_SPECTRO_REDUX'] = '/global/cfs/cdirs/desi/spectro/redux/andes'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Run balfinder on DESI QSO Catalog for Andes""")

parser.add_argument('-q','--qsocatalog', type = str, default = None, required = True,
                    help = 'Input QSO catalog name, e.g. qsocat-minisv.fits')

parser.add_argument('-o','--outdir', type=str, default=".", required=False,
                    help='Path for output BAL catalog')

parser.add_argument('-b','--balcatalog', type=str, default="balcat.fits",
                    required=False, help='Output BAL catalog name')

parser.add_argument('-c','--clobber', type=bool, default=False, required=False,
                    help='Clobber (overwrite) BAL catalog if it already exists?')

parser.add_argument('-v','--verbose', type=bool, default=False, required=False,
                    help='Provide verbose output?')

args  = parser.parse_args()

rootbaldir = args.outdir
qsocatname = args.qsocatalog
balcatname = rootbaldir + '/' + args.balcatalog

if debug: 
    args.clobber = True
    args.verbose = True

try: 
    qhdu = fits.open(qsocatname)
except FileNotFoundError:
    print("Error: ", qsocatname, "not found")
    exit(1)

if not os.path.exists(rootbaldir):
    os.makedirs(rootbaldir)

if args.verbose:
    print("Root output directory = ", rootbaldir)
    print("Output catalog name = ", balcatname)

# Identify unique tiles, dates in QSO catalog
tileids = list(set(qhdu[1].data['TILEID']))
print("unique tileids: ", tileids)

# DEBUG
# tileids = [68000]

pcaeigen = fitsio.read(bc.pcaeigenfile)

first = True

# For each tileid, identify the night
for tileid in tileids:
    tilemask = qhdu[1].data['TILEID'] == tileid
    night = list(set(qhdu[1].data['NIGHT'][tilemask]))[0]

    # identify the unique spectrographs for that tileid/night
    nightmask = tilemask*(qhdu[1].data['NIGHT'] == night)
    sps = list(set(qhdu[1].data['PETAL_LOC'][nightmask]))
    # DEBUG
    # sps = [0, 1, 2]

    # For each SP, coadd the data, run balfinder
    for sp in sps:
        specdir = os.path.join(os.environ['DESI_SPECTRO_REDUX'], "tiles", str(tileid), str(night))
        specfile = specdir + '/coadd-' + str(sp) + '-' + str(tileid) + '-' + str(night) + '.fits'

        # Identify the QSOs in this coadd
        specmask = nightmask*(qhdu[1].data['PETAL_LOC'] == sp)
        print(tileid, night, sp, np.sum(specmask), specfile)

        # Identify TARGETIDs for balfinder
        targetids = qhdu[1].data['TARGETID'][specmask]

        # Redshift data to include in BAL catalog
        zdata = np.zeros(len(qhdu[1].data['Z'][specmask]), dtype={'names':('Z', 'ZERR', 'ZWARN'), 'formats':('>f8', '>f8', '>f8')})
        zdata['Z'] = qhdu[1].data['Z'][specmask]
        zdata['ZERR'] = qhdu[1].data['ZERR'][specmask]
        zdata['ZWARN'] = qhdu[1].data['ZWARN'][specmask]

        # Other data to include in BAL catalog
        specdata = np.zeros(len(qhdu[1].data['TARGETID'][specmask]), dtype={'names':('TARGETID', 'TARGET_RA', 'TARGET_DEC', 'NIGHT', 'EXPID', 'MJD', 'TILEID'), 'formats':('>i8', '>f8', '>f8', '>i8', '>i8', '>f8', '>i8')})
        specdata['TARGETID'] = qhdu[1].data['TARGETID'][specmask]
        specdata['TARGET_RA'] = qhdu[1].data['TARGET_RA'][specmask]
        specdata['TARGET_DEC'] = qhdu[1].data['TARGET_DEC'][specmask]
        specdata['NIGHT'] = qhdu[1].data['NIGHT'][specmask]
        specdata['EXPID'] = qhdu[1].data['EXPID'][specmask]
        specdata['MJD'] = qhdu[1].data['MJD'][specmask]
        specdata['TILEID'] = qhdu[1].data['TILEID'][specmask]

        specobj = desispec.io.read_spectra(specfile)
        print("Coadding ", specfile) 
        if 'brz' not in specobj.wave.keys():
            specobj = coadd_cameras(specobj, cosmics_nsig=None)
        fm = specobj.fibermap
        # Identify these QSOs in the coadd data
        qsos = np.zeros(len(specdata), dtype=int)
        for i in range(len(specdata)):
            qsos[i] = np.where(targetids[i] == fm['TARGETID'])[0][0]
        balfilename = 'baltmp.fits'
        baltable.initbaltab_desi(specdata, zdata, balfilename, overwrite=True)
        print("Initialized temporary BAL catalog ", balfilename," with ", len(specdata), "entries")
        balhdu = fits.open(balfilename)

        # Loop through the QSOs and run the BAL finder
        for i in range(len(qsos)):
            targetid = targetids[i]
            qso = qsos[i] # index in specobj for QSO targetid
            zspec = zdata['Z'][i]
            if zspec >= bc.BAL_ZMIN and zspec <= bc.BAL_ZMAX: 
                qsospec = np.zeros(len(specobj.wave['brz']),dtype={'names':('wave', 'flux', 'ivar', 'model'), 'formats':('>f8', '>f8', '>f8', '>f8')})
                qsospec['wave'] = specobj.wave['brz']
                qsospec['flux'] = specobj.flux['brz'][qso]
                qsospec['ivar'] = specobj.ivar['brz'][qso]
                qsospec['model'] = np.zeros(len(specobj.wave['brz'])) # add to match SDSS format
                info, pcaout, mask = fitbal.calcbalparams(qsospec, pcaeigen, zspec)
                # update baltable
                balhdu = baltable.updatebaltab_desi(targetid, balhdu, info, pcaout)
                if args.verbose:
                    print("{0} Processed {1} at z = {2:.2f}: AI_CIV = {3:.0f}, BI_CIV = {4:.0f}".format(i, targetid, zspec, info['AI_CIV'], info['BI_CIV']))
            else: 
                    print("Skipped ", targetid, " as z = ", zspec) 

        balhdu.writeto(balfilename, overwrite=True)
        if args.verbose:
            print("Wrote output BAL catalog ", balfilename)
        baltmptab = Table.read(balfilename) 

        if first: 
            baltab = baltmptab.copy()
            first = False
        else: 
            baltab = vstack([baltab, baltmptab])

baltab.write(balcatname, format='fits', overwrite=True)
lastupdate = "Last updated {0} UT by {1}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), os.getlogin())
commandline = " ".join(sys.argv)
fits.setval(balcatname, 'COMMENT', value=commandline, ext=1)
fits.setval(balcatname, 'HISTORY', value=lastupdate, ext=1)
if args.verbose:
    print("Wrote output BAL catalog ", balcatname)


