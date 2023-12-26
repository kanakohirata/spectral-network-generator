from copy import deepcopy
import json
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import os
import re


LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


def _convert_string_peaks_to_floats(peaks):
    peaks = list(map(lambda x: [float(x[0]), float(x[1])], peaks))
    peaks.sort(key=lambda x: x[0])

    return peaks


def _merge_peaks(base_peaks, array_of_peaks, intensity=None):
    base_mzs = [p[0] for p in base_peaks]
    base_intensities = [float(p[1]) for p in base_peaks]
    
    if intensity:
        intensity_to_add = intensity
    else:
        intensity_to_add = np.median(base_intensities)

    

    for peaks in array_of_peaks:
        for mz, intensity in peaks:
            if mz not in base_mzs:
                base_peaks.append((mz, intensity_to_add))
                base_mzs.append(mz)
    
    return base_peaks


def read_cfm_result(path, is_convert_energies):
    spectra = []
    spectrum_meta = {}
    peaks_energy0 = []
    peaks_energy1 = []
    peaks_energy2 = []
    list_of_peaks_to_export = []
    flag_energy = -1
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('#'):
                _parse_metadata(line, spectrum_meta)

            elif line == 'energy0':
                flag_energy = 0
            elif line == 'energy1':
                flag_energy = 1
            elif line == 'energy2':
                flag_energy = 2

            elif re.match(r'(\d+\.\d+) (\d+\.\d+)', line):
                mz, intensity = re.match(r'(\d+\.\d+) (\d+\.\d+)', line).groups()[:2]
                
                if flag_energy == 0:
                    peaks_energy0.append((mz, intensity))
                elif flag_energy == 1:
                    peaks_energy1.append((mz, intensity))
                elif flag_energy == 2:
                    peaks_energy2.append((mz, intensity))

            elif not line:
                break

        if spectrum_meta.get('Adduct'):
            if spectrum_meta['Adduct'] in ('[M+H]+','[M]+','[M+NH4]+','[M+Na]+','[M+K]+','[M+Li]+'):
                    spectrum_meta['Ion_Mode'] = 'Positive'
            elif spectrum_meta['Adduct'] in ('[M-H]-','[M]-','[M+Cl]-','[M+HCOOH-H]-','[M+CH3COOH-H]-','[M-2H]2-'):
                spectrum_meta['Ion_Mode'] = 'Negative'

        if is_convert_energies:
            peaks_energy2 = _merge_peaks(peaks_energy2, (peaks_energy0, peaks_energy1))
            list_of_peaks_to_export.append((peaks_energy2, 'merged_energy'))
        else:
            list_of_peaks_to_export.extend([(peaks_energy0, 'energy0'), (peaks_energy1, 'energy1'), (peaks_energy2, 'energy2')])

        for peaks, energy in list_of_peaks_to_export:
            spectrum = deepcopy(spectrum_meta)
            peaks = _convert_string_peaks_to_floats(peaks)
            peaks_str = list(map(lambda x: f'[{x[0]},{x[1]}]', peaks))
            spectrum['energy'] = energy
            spectrum['peaks'] = peaks
            spectrum['peaks_json'] = '[' + ','.join(peaks_str) + ']'
            spectra.append(spectrum)

    return spectra


def convert_cfm_to_json(dir_path, output_path, is_convert_energies=True):
    LOGGER.debug('Convert CFM prediction files to a JSON file.')
    paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    if not paths:
        LOGGER.warning(f'No CFM prediction files in {dir_path}')
        return
    
    spectra = []

    for path in paths:
        _spectra = read_cfm_result(path, is_convert_energies=is_convert_energies)
        for spectrum in _spectra:
            spectrum = spectrum.pop('peaks')
            spectra.append(spectrum)

    with open(output_path, 'w') as j:
        json.dump(spectra, j, indent=4)

    print(1)


def _parse_metadata(line:str, data:dict):
    line = line.lstrip('#')
    if re.match(r'In-silico ESI-MS/MS (\[.+].+) Spectra', line):
        precursor_type = re.match(r'In-silico ESI-MS/MS (\[.+].+) Spectra', line).groups()[0]
        data['Adduct'] = precursor_type

    elif line.startswith('PREDICTED BY CFM-ID'):
        data['in_silico_tool'] = line
    
    elif line.startswith('ID='):
        id_ = line.replace('ID=', '')
        data['spectrum_id'] = id_
        data['accession'] = id_

    elif line.startswith('InChI='):
        data['INCHI'] = line
    
    elif line.startswith('InChiKey='):
        data['InChIKey'] = line.replace('InChiKey=', '')

