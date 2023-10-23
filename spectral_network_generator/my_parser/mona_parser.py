import json
from logging import DEBUG, Formatter, getLogger, StreamHandler
from matchms import Spectrum
import numpy as np
import re

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


def read_json(path, parameters_for_open=None):
    """Read a MoNa JSON file into a list of peaks and metadata.

    Parameters
    ----------
    path : str
    parameters_for_open : dict, optional
        This parameter will be passed to open(path).

    Returns
    -------
    list[tuple]
        A list of tuples of peaks and metadata.
        peaks: [[m/z, intensity], ...]
        metadata: {'metadata_key_1': value1, 'metadata_key_2': value2, ...}
    """
    if not isinstance(parameters_for_open, dict):
        parameters_for_open = {}
    
    LOGGER.info(f'Read {path}')
    with open(path, 'r', **parameters_for_open) as f:
        mona_spectra = json.load(f)
        print(1)

        parsed_spectra = []
        for s in mona_spectra:
            peaks = _parse_spectrum(s['spectrum'])
            if not peaks:
                continue
            metadata = _parse_metadata(s['metaData'])
            metadata.update(_parse_compound(s['compound'][0]))
            metadata['id'] = s.get('id', '')
            parsed_spectra.append((peaks, metadata))

    return parsed_spectra


def _parse_spectrum(spectrum):
    if isinstance(spectrum, list):
        return spectrum

    elif isinstance(spectrum, str):
        peaks = []
        if re.search(r'(\d+\.?\d+:\d+\.?\d+)', spectrum):
            str_peaks = re.findall(r'(\d+\.?\d+:\d+\.?\d+)', spectrum)
            
            for p in str_peaks:
                mz, intensity = p.split(':')
                peaks.append([float(mz), float(intensity)])

            return peaks
        
        elif re.search(r'(\d+\.?\d+,\s?\d+\.?\d+)', spectrum):
            str_peaks = re.findall(r'(\d+\.?\d+,\s?\d+\.?\d+)', spectrum)
            for p in str_peaks:
                mz, intensity = p.split(',')
                peaks.append([float(mz), float(intensity)])
        
            return peaks
        
    return peaks



def _parse_compound(compound:dict):
    name = ''
    for n in compound['names']:
        if n['name']:
            if not name:
                name = n['name']
            else:
                name += f'|{n["name"]}'
    LOGGER.debug(name)
    parsed = {'compound_name': name}

    # Get metadata
    for metadata in compound['metaData']:
        if metadata['name'] == 'InChI':
            parsed['inchi'] = metadata.get('value', '')

        elif metadata['name'] == 'InChIKey':
            parsed['inchikey'] = metadata.get('value') or compound.get('inchikey', '')
        
        elif metadata['name'] == 'molecular formula':
            parsed['molecular_formula'] = metadata.get('value', '')

        elif metadata['name'] == 'SMILES':
            parsed['smiles'] = metadata.get('value', '')
        
        elif metadata['name'] == 'total exact mass':
            parsed['exact_mass'] = metadata.get('value', 0)

    # Get classification
    for c in compound['classification']:
        if not c['value']:
            continue

        if c['name'] == 'pathway':
            LOGGER.warning(parsed['compound_name'])

        level = 'classification_' + c['name'].replace(' ', '_')
        if level not in parsed:
            parsed[level] = c['value']
        else:
            parsed[level] += f'|{c["value"]}'

    return parsed
    

def _parse_metadata(metadata:list):
    parsed = {}
    for data in metadata:
        k = data['name']
        v = data['value']
        k = k.lower().replace(' ', '_')
        if k != 'compound_id':
            try:
                v = float(v)
            except ValueError:
                pass
        parsed[k] = v

    # Fix 'fragmentation_type' value
    if not parsed.get('fragmentation_type'):
        if 'fragmentation_mode' in parsed or 'fragmentation_method' in parsed:
            parsed['fragmentation_type'] = parsed.get('fragmentation_mode') or parsed.get('fragmentation_method', '')

    if not parsed.get('fragmentation_type') and parsed.get('collision_energy'):
        parsed['fragmentation_type'] = 'CID'

    # Fix 'ion_source' value
    if parsed.get('ion_source') in ('LC-ESI', 'DI-ESI'):
        parsed['ion_source'] = 'ESI'
        
    return parsed

    
def convert_json_to_matchms_spectra(path, parameters_for_open=None):
    """Read a MoNa JSON file and create a list of matchms.Spectrum.

    Parameters
    ----------
    path : str
    parameters_for_open : dict, optional
        This parameter will be passed to open(path).

    Returns
    -------
    list[matchms.Spectrum]
    """
    matchms_spectra = []
    parsed_spectra = read_json(path, parameters_for_open)

    for peaks, metadata in parsed_spectra:
        if not peaks:
            continue
        
        mz_values = np.array(peaks)[:, 0]
        intensities = np.array(peaks)[:, 1]

        if not np.all(mz_values[:-1] <= mz_values[1:]):
            idx_sorted = np.argsort(mz_values)
            mz_values = mz_values[mz_values]
            intensities = intensities[idx_sorted]
        
        matchms_spectra.append(
            Spectrum(mz=mz_values,
                    intensities=intensities,
                    metadata=metadata,
                    metadata_harmonization=True)
        )

    return matchms_spectra
