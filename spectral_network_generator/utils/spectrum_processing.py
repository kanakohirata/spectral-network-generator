import logging
from logging import DEBUG, Formatter, getLogger, StreamHandler
from matchms import Spectrum
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
    new_spectrum = Spectrum(mz=mz_arr, intensities=intensity_arr)

    return new_spectrum






    
