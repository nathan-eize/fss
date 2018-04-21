#!/usr/bin/env python
"""
  Core script and library to perform Fractional Skill Score calculation on the BARRA netCDF files with gridded obs like TRMM and AWAP.

  Nathan Eizenberg, April 2018 adapted from Peter Steinle's original fss3.py code
"""

import os, sys
import argparse
from datetime import datetime as dt
from datetime import timedelta as delt
import numpy as np
from glob import glob
from netCDF4 import Dataset, num2date

sys.path.append('~/.local/lib/python2.7/site-packages')
import xarray as xr
#import iris

#iris.FUTURE.netcdf_promote = True

# Directory for trmm core methods
assert os.path.exists(os.getcwd() + '/../trmm_imerg/trmm_core2.py'), "Needs to find ../trmm_imerg/trmm_core2.py"
sys.path.append(os.path.join(os.getcwd(),'../trmm_imerg'))
import trmm_core2 as trmm

# Directory for awap core methods
assert os.path.exists(os.getcwd() + '/../awap'), "Needs to find ../awap"
sys.path.append(os.path.join(os.getcwd(),'../awap'))
import awap_core as awap

import fss_core as fss

#--------------------------------------------
# Define the argument options
#--------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-m",
            "--model",
            nargs="?",
	    type=str,
            help="Model data to use, eg. BARRA_R, BARRA_SY, etc...")
#parser.add_argument("-d",
#            "--date",
#            nargs="?",
#	    type=str,
#            help="Start date to perform 24 hour FFS, ISO8601 format eg. 20140714T1800Z")
parser.add_argument("-o",
            "--obstype",
            nargs="?",
            type=str,
            help="Type of observational data to use, eg. trmm, awap")
parser.add_argument("-i",
            "--infile",
            nargs="?",
            type=str,
            help="Model input file on the same grid as the obs")

#--------------------------------------------
# Misc definitions 
#--------------------------------------------

def stdout(s):
    "Prints a formatted message to standardout"
    print("[{}] >>>\t{}".format(sys.argv[0], s))

_iso_format = '%Y%m%dT%H%MZ' # ISO 8601 format
trmm_hours_total = int(3)
trmmBaseHours = range(0,24,3)
barra2trmm_dt_map = lambda dt: dt - delt(hours=1,minutes=30)

def merge_data( fid ):
    """ There is a problem with the interpolation where the time dimension is split into three iseperate dimensions 'time', 'time0' and 'time1'. This merges the three and produces a data xarra that is in order
    """
    # Load xarray, not that forecast_period only has one repeat
    mdat = xr.DataArray( 
	        np.concatenate( [ fid.variables[var][:] for var in ['accum_prcp', 'accum_prcp_0', 'accum_prcp_1'] ]), \
	        coords = {
	    'time': np.concatenate([num2date(fid.variables[tvar][:], fid.variables['time'].units) for tvar in ['time','time_0', 'time_1'] ]),\
	    #'forecast_period': np.concatenate([fid.variables[tvar][:] for tvar in ['forecast_period','forecast_period', 'forecast_period_0'] ]),\
	    'latitude': fid.variables['trmm_lat'][:],\
	    'longitude': fid.variables['trmm_lon'][:]},\
                dims = ['time', 'latitude','longitude']
		)

    return mdat.sortby('time')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main programs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

stdout('TEST CONFIG'.format(sys.argv[0]))
stdout('')
args = parser.parse_args()
stdout('ARGS:')
stdout(args)
stdout('')

# get model data
modData = {}
if not args.model.startswith('BARRA'):
    stdout('Not yet written for model data other than BARRA')
    sys.exit(0)
else:
    stdout('Using {} for model data'.format(args.model))
    assert os.path.exists(args.infile), "Cannot find file {}".format(args.infile)
    mod_fid = Dataset(args.infile, 'r')
    if all([ v in mod_fid.variables.keys() for v in ['time','time_0', 'time_1']]):
	mod_dat = merge_data(mod_fid)
    else:
	print('Not yet written')
    mod_fid.close()

modData.update({
		'type':args.model,\
	       	'prcp':mod_dat.data,\
		'lat':mod_dat.latitude.data,\
		'lon':mod_dat.longitude.data
		})

assert 'time' in mod_dat.coords.keys(), "Model data is not read in properly, requires time dim"

# get obs data
obsData = {}
if args.obstype == 'trmm':
    stdout('Using TRMM observational data')
    # get the hour that is closest to the trmm base hours
    #trmmHr = np.argmin(np.abs(dStart.hour - np.array(trmmBaseHours)))
    #stdout("Changing the start hour to {:d} to align with trmm obs".format(trmmHr))
    #stdout('New time start: {}'.format(dStart.strftime(_iso_format))) 
    #stdout('New time end: {}'.format(dEnd.strftime(_iso_format)))
    stdout('')
    # grab the right trmm data grids using the trmm_core modules
    #obs_dat, _,= trmm._get_single_grid(mod_dat.time.data[0])
    obs_dat, obs_err, obs_head_dict = trmm._get_single_grid(barra2trmm_dt_map(mod_dat.time.data[0]), get_header_info=True)
    obs_dat, _, obs_lat, obs_lon = trmm._to_grid(obs_dat, obs_err)
    stdout('')
else:
    stdout('Error, obs type {} not yet written yet'.format(args.obstype))
    sys.exit(0)

obsData.update({
		'type':args.obstype,\
	       	#'prcp':np.rollaxis(obs_dat,2,start=0),\
	       	'prcp':obs_dat,\
		'lat':obs_lat,\
		'lon':obs_lon
		})


# Now limit the lat lon obs grids to fit the model data
obsLonMask = np.logical_and(obsData['lon']<=modData['lon'].max(), obsData['lon']>=modData['lon'].min()) 
obsLonMaskDim = obsLonMask.sum()
obsLatMask = np.logical_and(obsData['lat']<=modData['lat'].max(), obsData['lat']>=modData['lat'].min()) 
obsLatMaskDim = obsLatMask.sum()
obsLonLonMask, obsLatLatMask = np.meshgrid(obsLonMask, obsLatMask)
obsPrcpMask = np.logical_and(obsLonLonMask, obsLatLatMask)
gridSize = obsPrcpMask.sum()
obsData.update({'lat':obsData['lat'][obsLatMask],\
	       	'lon':obsData['lon'][obsLonMask],\
		'prcp':obsData['prcp'][obsPrcpMask].reshape((obsLatMaskDim,obsLonMaskDim))
		})

## Interpolate function to the obs grid
#obsLatLat, obsLonLon = np.meshgrid(obsData['lat'], obsData['lon'])
#obsCoords = np.rollaxis(np.vstack((obsLonLon.reshape(-1), obsLatLat.reshape(-1))),1,0)
##interpNN = interpolate.NearestNDInterpolator(obsCoords,obsData['prcp'].reshape(-1))
#modLatLat, modLonLon = np.meshgrid(modData['lat'], modData['lon'])
#modCoords = np.rollaxis(np.vstack((modLonLon.reshape(-1), modLatLat.reshape(-1))),1,0)
#interpNNMod = interpolate.NearestNDInterpolator(modCoords,modData['prcp'].reshape(-1))
## interpolate
#modData['prcp'] = interpNNMod( obsCoords ).reshape((obsLatMaskDim,obsLonMaskDim))


# test the fractional skill score calcs
thresh = float(0.4)
maxDist = min(obsData['lat'].max() - obsData['lat'].min(), obsData['lon'].max() - obsData['lon'].min())
grid_space = ((obsData['lat'][1]-obsData['lat'][0]) + (obsData['lon'][1]-obsData['lon'][0]))/2
max_scan = maxDist / grid_space

nres = [ int(10**x) for x in np.arange(np.log10(max_scan/6.0), np.log10(max_scan), np.log10(1.4))]
nres.insert(0,5)
resArr = np.array( [ i if i%2 != 0 else i + 1 for i in nres], dtype=int)

numDf, denomDf, fssDf = fss.fss_frame(modData['prcp'][0], obsData['prcp'], nres, [thresh])

stdout('Done')
