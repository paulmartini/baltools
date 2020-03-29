========
baltools
========

Introduction
------------

Tools to identify, characterize, and simulate BAL QSOs. 


Installation
------------

These tools include dependencies to several DESI packages, and it should be most straightforward to use them at NERSC. 

Manually install them from the git checkout and then add the "py" directory to your $PYTHONPATH environment variable. 

Getting Started
---------------

See the notebook "DESI balfinder tutorial" in the "doc" directory for examples of how to run the balfinder on SV data and on mocks. 


Overview
--------

Here is a brief description of the files in the "py" directory:

**balconfig.py** : common modules and global variables used by balfinder

**baltable.py** : routines to construct BAL catalogs

**desibal.py** : main balfinder for DESI

**fitbal.py** : core code to identify and characterize BALs

**plotter.py** : various convenience plotting routines

**utils.py** : utilities for manipulating SDSS spectra
