from glob import glob
import h5py
import logging
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
from matchms.filtering import (add_retention_time, add_retention_index, derive_inchi_from_smiles,
                               derive_inchikey_from_inchi, normalize_intensities, select_by_relative_intensity)
from matchms.importing import load_from_json, load_from_mgf, load_from_msp
import os
import pickle
import re
import time
from my_parser.mona_parser import convert_json_to_matchms_spectra
from utils.spectrum_processing import deisotope, introduce_random_delta_to_mz, set_top_n_most_intense_peaks_in_bin

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


def delete_serialize_spectra_file():
    for p in glob('./serialized_spectra/*.pickle'):
        os.remove(p)

    for p in glob('./serialized_spectra/filtered/*.pickle'):
        os.remove(p)


def load_and_serialize_spectra(spectra_path, dataset_tag, intensity_threshold=0.001, is_introduce_random_mass_shift=False,
                               deisotope_int_ratio=-1, deisotope_mz_tol=0, binning_top_n=0, binning_range=-1, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    last_index = -1
    with h5py.File('spectrum_metadata.h5', 'r') as h5:
        if 'metadata' in h5.keys():
            last_index = h5['metadata'].len() - 1

    serialized_spectra_path_to_rename = ''
    if last_index != -1 and float(last_index / 1000) * 1000 != 999:
        _old_serialized_spectra_idx_start = int(max(int(last_index / 1000) * 1000, 0))
        _old_serialized_spectra_idx_end = last_index
        serialized_spectra_path_to_rename = f'./serialized_spectra/{_old_serialized_spectra_idx_start}' \
                                            f'-{_old_serialized_spectra_idx_end}.pickle'

    logger.info(f'Read spectral data: {spectra_path}')

    spectrum_metadata_list = []
    _count = 0
    spectra_filename = os.path.basename(spectra_path)
    logger.info(f'Filename: {spectra_filename}')

    if spectra_filename.endswith('.msp'):
        logger.info('Load from msp')
        spectra_file = load_from_msp(spectra_path)
    elif spectra_filename.endswith('.mgf'):
        logger.info('Load from mgf')
        spectra_file = load_from_mgf(spectra_path)
    elif spectra_filename.endswith('.json'):
        logger.info('Load from json')
        spectra_file = load_from_json(spectra_path)

        if not spectra_file:
            spectra_file = convert_json_to_matchms_spectra(spectra_path)
    else:
        return

    if not spectra_file:
        return

    if is_introduce_random_mass_shift:
        logger.info(f'Introduce random mass shift: {spectra_filename}')
    
    _spectra = []
    for _s in spectra_file:
        last_index += 1
        _s.set('index', last_index)

        if is_introduce_random_mass_shift:
            _s = introduce_random_delta_to_mz(_s, 0, 50)

        if deisotope_int_ratio > 0:
            _s = deisotope(_s, deisotope_mz_tol, deisotope_int_ratio)

        if binning_range > 0:
            _s = set_top_n_most_intense_peaks_in_bin(_s, binning_top_n, binning_range)

        _s = normalize_intensities(_s)
        _s = add_retention_time(_s)
        _s = add_retention_index(_s)
        _s = select_by_relative_intensity(_s, intensity_from=intensity_threshold)

        _rt_in_sec = 0
        if _s.get('retention_time'):
            _rt_in_sec = _s.get('retention_time')
        elif _s.get('rtinminutes'):
            _rt_in_sec = float(_s.get('rtinminutes')) * 60

        if not _s.get('smiles') and _s.get('computed_smiles'):
            _s.set('smiles', _s.get('computed_smiles'))
        if not _s.get('inchi') and _s.get('computed_inchi'):
            _s.set('inchi', _s.get('computed_inchi'))

        _s = derive_inchi_from_smiles(_s)
        _s = derive_inchikey_from_inchi(_s)
        if _s.get('accession_number'):
            pass
        elif _s.get('accession'):
            _s.set('accession_number', _s.get('accession'))
            _s.set('global_accession', f'{_count}|{_s.get("accession")}|{spectra_filename}')
        elif _s.get('db#'):
            _s.set('accession_number', _s.get('db#'))
            _s.set('global_accession', f'{_count}|{_s.get("db#")}|{spectra_filename}')
        elif _s.get('title'):
            _s.set('accession_number', _s.get('title'))
            _s.set('global_accession', f'{_count}|{_s.get("title")}|{spectra_filename}')
        else:
            _s.set('accession_number', str(_count))
            _s.set('global_accession', f'{_count}|{spectra_filename}')

        if _s.get('title'):
            pass
        else:
            _s.set('title', str(_count))
        _s.set('source_filename', spectra_filename)

        str_peaks = ', '.join(map(lambda x: f'[{x[0]}, {x[1]}]', _s.peaks))
        str_peaks = '[' + str_peaks + ']'
        str_mz = '[' + ', '.join(map(lambda x: str(x), _s.mz)) + ']'

        # if _s.get('kingdom'):
        #     _k = _s.get('kingdom')
        #     logger.debug(_k)
        # if _s.get('alternative_parent'):
        #     _a = _s.get('alternative_parent')
        #     logger.debug(_a)

        spectrum_metadata_list.append((
            last_index,
            dataset_tag,
            spectra_filename,
            _s.get('global_accession', ''),
            _s.get('accession_number', ''),
            _s.get('precursor_mz', 0),
            _rt_in_sec,
            _s.get('retention_index', 0) or 0,
            _s.get('inchi', ''),
            _s.get('inchikey', ''),
            _s.get('author', ''),
            _s.get('compound_name', ''),
            _s.get('title', ''),
            _s.get('instrument_type', ''),
            _s.get('ionization_mode') or _s.get('ionization') or _s.get('ion_mode') or _s.get('ionmode', ''),
            _s.get('precursor_type', ''),
            _s.get('fragmentation_type') or _s.get('fragmentation_mode') or _s.get('fragmentation', ''),
            _s.mz.size,
            str_peaks,
            str_mz,
            '', '', '',
            _s.get('classification_superclass', ''),
            _s.get('classification_class', ''),
            _s.get('classification_subclass', ''),
            _s.get('classification_alternative_parent', ''),
        ))

        _spectra.append(_s)
        _count += 1

        if (last_index + 1) % 1000 == 0:
            _serialized_spectra_idx_start = last_index - 999
            _serialized_spectra_idx_end = last_index
            _serialized_spectra_path = f'./serialized_spectra/' \
                                       f'{_serialized_spectra_idx_start}-{_serialized_spectra_idx_end}.pickle'

            if serialized_spectra_path_to_rename:
                os.rename(serialized_spectra_path_to_rename, _serialized_spectra_path)
                serialized_spectra_path_to_rename = ''

            time.sleep(0.5)
            try:
                with open(_serialized_spectra_path, 'rb') as fr:
                    _spectra_old = pickle.load(fr)
            except (EOFError, FileNotFoundError):
                _spectra_old = []
            with open(_serialized_spectra_path, 'wb') as f:
                pickle.dump(_spectra_old + _spectra, f)
                f.flush()

            _spectra = []

    _serialized_spectra_idx_start = int(max(int(last_index / 1000) * 1000, 0))
    _serialized_spectra_idx_end = last_index
    serialized_spectra_path = f'./serialized_spectra/' \
                              f'{_serialized_spectra_idx_start}-{_serialized_spectra_idx_end}.pickle'
    if serialized_spectra_path_to_rename:
        os.rename(serialized_spectra_path_to_rename, serialized_spectra_path)
    try:
        with open(serialized_spectra_path, 'rb') as fr:
            _spectra_old = pickle.load(fr)
    except (EOFError, FileNotFoundError):
        _spectra_old = []
    with open(serialized_spectra_path, 'wb') as f:
        pickle.dump(_spectra_old + _spectra, f)
        f.flush()

    with h5py.File('spectrum_metadata.h5', 'a') as h5:
        _arr = np.array(spectrum_metadata_list,
                        dtype=[
                            ('index', 'u8'), ('tag', H5PY_STR_TYPE), ('source_filename', H5PY_STR_TYPE),
                            ('global_accession', H5PY_STR_TYPE), ('accession_number', H5PY_STR_TYPE),
                            ('precursor_mz', 'f8'), ('rt_in_sec', 'f8'),
                            ('retention_index', 'f8'), ('inchi', H5PY_STR_TYPE), ('inchikey', H5PY_STR_TYPE),
                            ('author', H5PY_STR_TYPE), ('compound_name', H5PY_STR_TYPE), ('title', H5PY_STR_TYPE),
                            ('instrument_type', H5PY_STR_TYPE), ('ionization_mode', H5PY_STR_TYPE),
                            ('fragmentation_type', H5PY_STR_TYPE), ('precursor_type', H5PY_STR_TYPE),
                            ('number_of_peaks', 'u8'), ('peaks', H5PY_STR_TYPE), ('mz_list', H5PY_STR_TYPE),
                            ('external_compound_unique_id_list', H5PY_STR_TYPE),
                            ('pathway_unique_id_list', H5PY_STR_TYPE),
                            ('pathway_common_name_list', H5PY_STR_TYPE),
                            ('cmpd_classification_superclass_list', H5PY_STR_TYPE),
                            ('cmpd_classification_class_list', H5PY_STR_TYPE),
                            ('cmpd_classification_subclass_list', H5PY_STR_TYPE),
                            ('cmpd_classification_alternative_parent_list', H5PY_STR_TYPE)])

        if 'metadata' not in set(h5.keys()):
            h5.create_dataset(name='metadata', data=_arr, shape=(_arr.shape[0],), maxshape=(None,))
        else:
            dset = h5['metadata']
            dset.resize((dset.len() + _arr.shape[0]), axis=0)
            dset[-_arr.shape[0]:] = _arr

        if dataset_tag != 'blank':
            if 'metadata' not in set(h5['filtered'].keys()):
                h5.create_dataset(name='filtered/metadata', data=_arr, shape=(_arr.shape[0],), maxshape=(None,))
            else:
                dset_filtered = h5['filtered/metadata']
                dset_filtered.resize((dset_filtered.len() + _arr.shape[0]), axis=0)
                dset_filtered[-_arr.shape[0]:] = _arr
        h5.flush()


def serialize_filtered_spectra():
    pickle_paths = glob('./serialized_spectra/*.pickle')
    with h5py.File('./spectrum_metadata.h5', 'r') as metadata_h5:
        indexes_to_calculate = metadata_h5['filtered/metadata'].fields('index')[()].astype(int)

    dump_count = 0
    count = 0
    spectra = []
    for pickle_path in pickle_paths:
        idx_start = int(re.findall(r'\d+', pickle_path)[0])
        idx_end = int(re.findall(r'\d+', pickle_path)[-1])
        _indexes_to_calculate = [i - idx_start for i in indexes_to_calculate if idx_start <= i <= idx_end]

        with open(pickle_path, 'rb') as f:
            _spectra = pickle.load(f)
            _spectra = list(map(lambda i: _spectra[i], _indexes_to_calculate))

            if count + len(_spectra) >= 1000 * (dump_count + 1):
                count_to_add = 1000 * (dump_count + 1) - count
                count += len(_spectra[:count_to_add])
                spectra += _spectra[:count_to_add]

                _new_spectra_idx_start = count - 1000
                _new_spectra_idx_end = count - 1
                _new_spectra_path = f'./serialized_spectra/filtered/' \
                                           f'{_new_spectra_idx_start}-{_new_spectra_idx_end}.pickle'

                with open(_new_spectra_path, 'wb') as f_new:
                    pickle.dump(spectra, f_new)
                    f_new.flush()

                count += len(_spectra[count_to_add:])
                spectra = _spectra[count_to_add:]
                dump_count += 1

            else:
                count += len(_spectra)
                spectra += _spectra

    _new_spectra_idx_start = int(count / 1000) * 1000
    _new_spectra_idx_end = count - 1
    _new_spectra_path = f'./serialized_spectra/filtered/' \
                        f'{_new_spectra_idx_start}-{_new_spectra_idx_end}.pickle'

    with open(_new_spectra_path, 'wb') as f_new:
        pickle.dump(spectra, f_new)
        f_new.flush()
