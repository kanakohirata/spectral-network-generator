import logging
from logging import DEBUG, Formatter, getLogger, StreamHandler
from matchms import Spectrum
import math
import numpy as np
import random

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


def deisotope(spectrum: Spectrum, mz_tolerance, intensiry_ratio, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    logger.debug('Deisotope')

    original_mz_arr = spectrum.mz
    original_intensity_arr = spectrum.intensities
    peak_length = original_mz_arr.size

    deisotoped_mz_values = []
    deisotoped_intensity_values = []

    for n in range (peak_length):
        flag_to_be_removed = False

        idx = peak_length - n - 1
        current_mz = original_mz_arr[idx]
        current_intensity = original_intensity_arr[idx]

        for extend in range(1, 11):
            if idx - extend < 0:
                break
            if original_mz_arr[idx - extend] < current_mz - 2:
                break
            
            mz_current_target = math.fabs(original_mz_arr[idx - extend] - current_mz + 1)
            if (mz_current_target < mz_tolerance
                and original_intensity_arr[idx - extend] >  current_intensity * intensiry_ratio):
                flag_to_be_removed =  True
                break

        if not flag_to_be_removed:
            deisotoped_mz_values.append(current_mz)
            deisotoped_intensity_values.append(current_intensity)
        
    deisotoped_mz_arr = np.array(deisotoped_mz_values)
    deisotoped_intensity_arr = np.array(deisotoped_intensity_values)
    sorted_idx_arr = np.argsort(deisotoped_mz_arr)
    deisotoped_mz_arr = deisotoped_mz_arr[sorted_idx_arr]
    deisotoped_intensity_arr = deisotoped_intensity_arr[sorted_idx_arr]

    new_spectrum = Spectrum(mz=deisotoped_mz_arr, intensities=deisotoped_intensity_arr, metadata=spectrum.metadata)
    return new_spectrum



def introduce_random_delta_to_mz(spectrum: Spectrum, start:int, end:int, _logger=None):
    """Introduce random mass shift to m/z values.

    Parameters
    ----------
    spectrum : matchms.Spectrum
    start : int
        This parameter will be passed to random.randint as first argument.
    end : int
        This parameter will be passed to random.randint as second argument.

    Returns
    -------
    matchms.Spectrum
    """
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER
    
    logger.debug('Introduce random mass shift to m/z values')
    mz_values = []
    for mz in spectrum.mz:
        random_delta = random.randint(start , end)
        mz_values.append(mz + random_delta)

    mz_arr = np.array(mz_values)
    sorted_indices = np.argsort(mz_arr)
    mz_arr = mz_arr[sorted_indices]
    intensity_arr = spectrum.intensities[sorted_indices]
    new_spectrum = Spectrum(mz=mz_arr, intensities=intensity_arr, metadata=spectrum.metadata)

    return new_spectrum   
