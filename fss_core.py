#!/usr/bin/env python
"""
  Core script and library to perform Fractional Skill Score calculation on the BARRA netCDF files with gridded obs like TRMM and AWAP.

  Nathan Eizenberg, April 2018 adapted from Peter Steinle's original fss3.py code
"""
#--------------------------------------------
# FSS definitions
#--------------------------------------------
# Added content
# Code from Faggian N, Roux B, Steinle P, Ebert B. 2015. Fast calculation of the fractions skill score. Mausam 66: 457466.

import collections
import scipy.signal as signal
import numpy as np
import pandas as pd

# Define an integral image data-type.
IntegralImage = collections.namedtuple(
    'IntegralImage',
    'table padding'
    )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def summedAreaTable(field, padding=None):
    """
    Code from Faggian N, Roux B, Steinle P, Ebert B. 2015. Fast calculation of the fractions skill score. Mausam 66: 457466.
    Returns the summed-area-table of the provided field.
    """

    if padding is None:
        return IntegralImage(
            table=field.cumsum(1).cumsum(0),
            padding=0
            )
    else:
        # Zero pad and account for windows centered on boundaries
        expandedField = np.zeros(
            (field.shape[0] + (2 * padding),
             field.shape[1] + (2 * padding))
            )

        expandedField[
            padding:padding + field.shape[0],
            padding:padding + field.shape[1]] = field

        # Compute the summed area table.
        return IntegralImage(
            table=expandedField.cumsum(1).cumsum(0),
            padding=padding
            )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def integralField(field, n, integral=None):
    """
    Code from Faggian N, Roux B, Steinle P, Ebert B. 2015. Fast calculation of the fractions skill score. Mausam 66: 457466.
    Fast summed area table version of the sliding accumulator.

    @param field: nd-array of binary hits/misses.
    @param n: window size.
    """

    window = n / 2

    if integral is not None:

        assert integral.padding >= window, 'Expected larger table.'

        integral = IntegralImage(
            table=integral.table[
               (integral.padding - window):-(integral.padding - window),
               (integral.padding - window):-(integral.padding - window)
               ],
            padding=window
            )

    else:
        integral = summedAreaTable(field, padding=window)

    # Compute the coordinates of the grid, offset by the window size.
    gridcells = np.mgrid[0:field.shape[0], 0:field.shape[1]] + integral.padding

    tl = (gridcells - integral.padding) - 1
    br = (gridcells + integral.padding - 1)

    sumField = integral.table[br[0], br[1]] + \
        integral.table[tl[0], tl[1]] - \
        integral.table[tl[0], br[1]] - \
        integral.table[br[0], tl[1]]

    # Fix values on the top and left boundaries of the field.
    sumField[:, 0] = integral.table[br[0], br[1]][:, 0]
    sumField[0, :] = integral.table[br[0], br[1]][0, :]
    sumField[integral.padding:, 0] -= \
        integral.table[tl[0], br[1]][integral.padding:, 0]
    sumField[0, integral.padding:] -= \
        integral.table[br[0], tl[1]][0, integral.padding:]

    return sumField

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fss(fcst, obs, threshold, window, fcst_cache=None, obs_cache=None):
    """
    Code from Faggian N, Roux B, Steinle P, Ebert B. 2015. Fast calculation of the fractions skill score. Mausam 66: 457466.
    Compute the fraction skill score.

    @param fcst: nd_array, forecast field.
    @param obs: nd_array, observation field.
    @param method: python function, defining the smoothing method.
    @return: float, numerator, denominator and FSS
    FSS = 1 - numerator/denominator
    """

    #fhat = integralField(fcst>threshold, window, integral=fcst_cache)
    #ohat = integralField(obs>threshold, window, integral=obs_cache)
    fhat = integralField(fcst>threshold, window)
    ohat = integralField(obs>threshold, window)

    scale = 1.0 / fhat.size
    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2))

    return num, denom, 1.-num/denom

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fss_frame(fcst, obs, windows, levels):
    """
    Faggian N, Roux B, Steinle P, Ebert B. 2015. Fast calculation of the fractions skill score. Mausam 66: 457466.
    Compute the fraction skill score data-fram.
    :param fcst: nd-array, forecast field.
    :param ods: nd-array, observation field.
    :param window: list, window sizes.
    :param levels: list, threshold levels.
    :return: list, dataframes of the fss: numerator,denominator and score.
    """
    num_data, den_data, fss_data = [], [], []

    for level in levels:
        ftable = summedAreaTable(fcst>level)
        otable = summedAreaTable(obs>level)
        #ftable = compute_integral_table(fcst > level)
	#otable = compute_integral_table(obs>level)

        _data = [fss(fcst, obs, level, w, fcst_cache=ftable, obs_cache=otable) for w in windows]

	num_data.append([x[0] for x in _data])
	den_data.append([x[1] for x in _data])
	fss_data.append([x[2] for x in _data])

    return (pd.DataFrame(num_data, index=levels, columns=windows),
           pd.DataFrame(den_data, index=levels, columns=windows),
           pd.DataFrame(fss_data, index=levels, columns=windows))
