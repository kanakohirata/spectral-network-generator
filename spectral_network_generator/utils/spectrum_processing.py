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


def deisotope(spectrum: Spectrum, mz_tolerance, intensity_ratio, _logger=None):
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
                and original_intensity_arr[idx - extend] >  current_intensity * intensity_ratio):
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

    return Spectrum(mz=deisotoped_mz_arr, intensities=deisotoped_intensity_arr, metadata=spectrum.metadata)


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

    return Spectrum(mz=mz_arr, intensities=intensity_arr, metadata=spectrum.metadata)


def set_intensity_in_log1p(spectrum:Spectrum, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    logger.debug('Convert intensity X to log(1 + X)')
    intensity_values = list(map(math.log1p, spectrum.intensities))
    return Spectrum(mz=spectrum.mz, intensities=np.array(intensity_values), metadata=spectrum.metadata)


def set_top_n_most_intense_peaks(spectrum:Spectrum, top_n:int, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    logger.debug(f'Keep {top_n} most intense peaks')

    if spectrum.mz.size <= top_n:
        return spectrum
    
    sorted_indices = np.argsort(spectrum.intensities)[::-1]

    mz_arr = spectrum.mz[sorted_indices][:top_n]
    intensity_arr = spectrum.intensities[sorted_indices][:top_n]

    sorted_indices = np.argsort(mz_arr)
    mz_arr = mz_arr[sorted_indices]
    intensity_arr = intensity_arr[sorted_indices]

    return Spectrum(mz=mz_arr, intensities=intensity_arr, metadata=spectrum.metadata)


def set_top_n_most_intense_peaks_in_bin(spectrum:Spectrum, top_n:int, bin_range:float, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    logger.debug(f'Keep {top_n} most intense peaks with bins of {bin_range} m/z')
    if spectrum.mz.size <= top_n:
        return spectrum
    
    top_n_peaks = []

    highest_mz = spectrum.mz[-1]
    n = 0
    list_mz_segment = []
    while n * bin_range <= highest_mz:
        list_mz_segment.append(n * bin_range)
        n += 1
    list_mz_segment.append(highest_mz)

    for n in range(1, len(list_mz_segment)):
        list_peak_in_bin = []
        # scan input peak list
        for peak in spectrum.peaks:
            if list_mz_segment[n-1] < peak[0] <= list_mz_segment[n]:
                list_peak_in_bin.append(peak)
            elif list_mz_segment[n] < peak[0]:
                break

        if len(list_peak_in_bin) > top_n:
            # reverse sort by intensity
            list_peak_in_bin.sort(key=lambda e:e[1], reverse=True)
            list_peak_in_bin = list_peak_in_bin[:top_n]
            list_peak_in_bin.sort(key=lambda e:e[0])

        top_n_peaks.extend(list_peak_in_bin)

    mz_values = [p[0] for p in top_n_peaks]
    intensity_values = [p[1] for p in top_n_peaks]

    mz_arr = np.array(mz_values)
    intensity_arr = np.array(intensity_values)

    return Spectrum(mz=mz_arr, intensities=intensity_arr, metadata=spectrum.metadata)
