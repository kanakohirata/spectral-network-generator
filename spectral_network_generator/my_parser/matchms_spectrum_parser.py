from glob import glob
import h5py
import logging
from logging import DEBUG, Formatter, getLogger, StreamHandler
from matchms import Spectrum
from matchms.filtering import (add_retention_index, derive_inchi_from_smiles,
                               derive_inchikey_from_inchi, normalize_intensities, select_by_relative_intensity)
from matchms.importing import load_from_json, load_from_mgf, load_from_msp
import os
import pickle
import re
import shutil
from my_parser.mona_parser import convert_json_to_matchms_spectra
from utils.spectrum_processing import (deisotope, introduce_random_delta_to_mz,
                                       set_top_n_most_intense_peaks,
                                       set_top_n_most_intense_peaks_in_bin)

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


def initialize_serialize_spectra_file(dir_path='./serialized_spectra'):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    else:
        for f in os.listdir(dir_path):
            p = os.path.join(dir_path, f)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.splitext(f)[1] == '.pickle':
                os.remove(p)

    dir_path_raw = os.path.join(dir_path, 'raw')
    if not os.path.isdir(dir_path_raw):
        os.makedirs(dir_path_raw)

    dir_path_filtered = os.path.join(dir_path, 'filtered')
    if not os.path.isdir(dir_path_filtered):
        os.makedirs(dir_path_filtered)

    dir_path_grouped = os.path.join(dir_path, 'grouped')
    if not os.path.isdir(dir_path_grouped):
        os.makedirs(dir_path_grouped)


def _convert_str_to_float(string):
    if not isinstance(string, str):
        return string

    try:
        return float(re.search(r'\d*\.?\d+', string).group())
    except TypeError:
        return 0.0


def set_retention_time_in_sec(spectrum: Spectrum, convert_min_to_sec: bool):
    """
    Set 'retention_time' and 'rt_in_sec' information.
    Parameters
    ----------
    spectrum : matchms.Spectrum
    convert_min_to_sec : bool
        If convert_min_to_sec is True, retention time is considered to be in minutes and is converted to seconds.

    Returns
    -------
    """
    retention_time_keys = ['retention_time', 'retentiontime', 'rt', 'scan_start_time', 'rt_query']
    rt = 0.0

    if spectrum.get('rtinseconds'):
        rt = _convert_str_to_float(spectrum.get('rtinseconds'))
    elif spectrum.get('rtinminutes'):
        rt = _convert_str_to_float(spectrum.get('rtinminutes')) * 60
    else:
        for k in retention_time_keys:
            if spectrum.get(k):
                rt = _convert_str_to_float(spectrum.get(k))

                if convert_min_to_sec:
                    rt = rt * 60

                break

    spectrum.set('retention_time', rt)
    spectrum.set('rt_in_sec', rt)


def repair_ion_mode(spectrum: Spectrum):
    """
    Correct ion mode. If it is 'pos' or 'p', it is changed to 'positive'. ('neg' and 'n' -> 'negative')
    If 'ionization_mode' has ion mode information, set the value to 'ionmode'.
    Parameters
    ----------
    spectrum : matchms.Spectrum

    Returns
    -------

    """
    ionization_mode = spectrum.get('ionization_mode') or spectrum.get('ionizationmode', '')
    ionization_mode = ionization_mode.lower()
    if ionization_mode in ('positive', 'pos', 'p', 'negative', 'neg', 'n'):
        spectrum.set('ionmode', ionization_mode)
        spectrum.set('ionization_mode', '')
        spectrum.set('ionizationmode', '')

    ionization = spectrum.get('ionization', '')
    ionization = ionization.lower()
    if ionization in ('positive', 'pos', 'p', 'negative', 'neg', 'n'):
        spectrum.set('ionmode', ionization)
        spectrum.set('ionization', '')

    ion_mode = spectrum.get('ion_mode') or spectrum.get('ionmode', '')
    if ion_mode in ('pos', 'p'):
        spectrum.set('ionmode', 'positive')
    elif ion_mode in ('neg', 'n'):
        spectrum.set('ionmode', 'negative')


def load_and_serialize_spectra(spectra_path, dataset_tag, serialized_spectra_dir, global_index_start,
                               intensity_threshold=0.001,
                               is_introduce_random_mass_shift=False,
                               deisotope_int_ratio=-1,
                               deisotope_mz_tol=0,
                               binning_top_n=0, binning_range=-1,
                               matching_top_n_input=-1,
                               _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    # If number of spectra is length_to_export, spectra is serialized.
    length_to_export = 1000

    # Make a folder if it does not exist.
    if not os.path.isdir(serialized_spectra_dir):
        os.makedirs(serialized_spectra_dir)

    # Get file paths in serialized_spectra_dir
    serialized_spectra_paths = get_serialized_spectra_paths(serialized_spectra_dir)
    if not serialized_spectra_paths:
        last_index = -1
    else:
        last_index = serialized_spectra_paths[-1][2]

    index = last_index + 1
    global_index = global_index_start

    serialized_spectra_path_to_update = ''
    # When there is a file containing fewer than length_to_export spectra, they will be added.
    if last_index != -1 and (last_index + 1) % length_to_export != 0:
        serialized_spectra_path_to_update = serialized_spectra_paths[-1]

    logger.info(f'Read spectral data: {spectra_path}')

    _count = 0
    spectra_filename = os.path.basename(spectra_path)
    logger.info(f'Filename: {spectra_filename}')

    if spectra_filename.endswith('.msp'):
        logger.info('Load from msp')
        spectra_file = load_from_msp(spectra_path)
        spectra_file_for_rt = load_from_msp(spectra_path)
    elif spectra_filename.endswith('.mgf'):
        logger.info('Load from mgf')
        spectra_file = load_from_mgf(spectra_path)
        spectra_file_for_rt = load_from_mgf(spectra_path)
    elif spectra_filename.endswith('.json'):
        logger.info('Load from json')
        spectra_file = load_from_json(spectra_path)
        spectra_file_for_rt = load_from_json(spectra_path)

        if not spectra_file:
            spectra_file = convert_json_to_matchms_spectra(spectra_path)
            spectra_file_for_rt = spectra_file
    else:
        return global_index

    if not spectra_file:
        return global_index

    if is_introduce_random_mass_shift:
        logger.info(f'Introduce random mass shift: {spectra_filename}')

    # Collect retention time ( not sure min or sec) ---------------------------------------------------
    # Obtain all retention times in advance and estimate whether they are in minutes or seconds.
    retention_time_keys = ['retention_time', 'retentiontime', 'rt', 'scan_start_time', 'rt_query']
    retention_time_list = []
    for _s in spectra_file_for_rt:
        rt = 0.0
        for k in retention_time_keys:
            if _s.get(k):
                rt = _s.get(k)
                if isinstance(rt, str):
                    try:
                        rt = float(re.search(r'\d*\.?\d+', rt).group())
                    except TypeError:
                        rt = 0.0
                break

        retention_time_list.append(rt)

    # we assume lc-run is about 5 min to 100 min
    # therefore, if max retention time is more than 100, it means RT is in sec.
    is_rt_in_min = False
    if max(retention_time_list) <= 100:
        is_rt_in_min = True
    # -------------------------------------------------------------------------------------------------

    # Add metadata to spectra and serialize them.
    _spectra = []
    for _s in spectra_file:
        _s.set('index', global_index)

        # Normalize intensities (Max = 1).
        _s = normalize_intensities(_s)

        # Retain peaks with intensity >= intensity_threshold.
        _s = select_by_relative_intensity(_s, intensity_from=intensity_threshold)

        if is_introduce_random_mass_shift:
            # Add random numbers to m/z values.
            _s = introduce_random_delta_to_mz(_s, 0, 50)

        if deisotope_int_ratio > 0:
            _s = deisotope(_s, deisotope_mz_tol, deisotope_int_ratio)

        if binning_range > 0:
            # Retain the top N intense peaks in each bin.
            _s = set_top_n_most_intense_peaks_in_bin(_s, binning_top_n, binning_range)

        if matching_top_n_input > 0:
            # Retain the top N intense peaks.
            _s = set_top_n_most_intense_peaks(_s, matching_top_n_input)

        # Set retention time in sec to 'retention_time' and 'rt_in_sec' of metadata.
        set_retention_time_in_sec(_s, is_rt_in_min)
        _s = add_retention_index(_s)

        # Add smiles and inchi
        if not _s.get('smiles') and _s.get('computed_smiles'):
            _s.set('smiles', _s.get('computed_smiles'))
        if not _s.get('inchi') and _s.get('computed_inchi'):
            _s.set('inchi', _s.get('computed_inchi'))

        _s = derive_inchi_from_smiles(_s)
        _s = derive_inchikey_from_inchi(_s)

        # Correct ion mode
        repair_ion_mode(_s)

        # Add accession number and global accession.
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

        # Add title
        if _s.get('title'):
            pass
        else:
            _s.set('title', str(_count))

        # Add source filename and tag.
        _s.set('source_filename', spectra_filename)
        _s.set('tag', dataset_tag)

        _spectra.append(_s)
        _count += 1
        global_index += 1

        # If number of spectra is length_to_export, spectra is serialized. ---------------------------------------
        if (index + 1) % length_to_export == 0:
            _serialized_spectra_idx_start = index - (length_to_export - 1)
            _serialized_spectra_idx_end = index
            _serialized_spectra_path = os.path.join(
                serialized_spectra_dir, f'{_serialized_spectra_idx_start}-{_serialized_spectra_idx_end}.pickle')

            # Get already serialized spectra
            _spectra_old = []
            if serialized_spectra_path_to_update:
                with open(serialized_spectra_path_to_update, 'rb') as fr:
                    _spectra_old = pickle.load(fr)

                # Remove a file containing fewer than length_to_export spectra.
                os.remove(serialized_spectra_path_to_update)
                serialized_spectra_path_to_update = ''

            with open(_serialized_spectra_path, 'wb') as f:
                _spectra = _spectra_old + _spectra

                if len(_spectra) != length_to_export:
                    raise ValueError(f'Number of spectra to serialize should be {length_to_export}')

                pickle.dump(_spectra, f)
                f.flush()

            # Initialize _spectra
            _spectra = []
        # --------------------------------------------------------------------------------------------------------

        index += 1

    _serialized_spectra_idx_start = int(max(int(index / length_to_export) * length_to_export, 0))
    _serialized_spectra_idx_end = index
    serialized_spectra_path = os.path.join(
        serialized_spectra_dir, f'{_serialized_spectra_idx_start}-{_serialized_spectra_idx_end}.pickle')

    # Get already serialized spectra
    _spectra_old = []
    if serialized_spectra_path_to_update:
        with open(serialized_spectra_path_to_update, 'rb') as fr:
            _spectra_old = pickle.load(fr)

        # Remove a file containing fewer than length_to_export spectra.
        os.remove(serialized_spectra_path_to_update)

    with open(serialized_spectra_path, 'wb') as f:
        _spectra = _spectra_old + _spectra
        pickle.dump(_spectra, f)
        f.flush()

    return global_index


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


def serialize_grouped_spectra():
    with h5py.File('./spectrum_metadata.h5', 'r') as metadata_h5:
        for dataset_keyword in metadata_h5['grouped'].keys():
            indexes_to_serialize = metadata_h5[f'grouped/{dataset_keyword}'].fields('index')[()].astype(int)

            output_dir = os.path.join('./serialized_spectra/grouped', dataset_keyword)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            dump_count = 0
            count = 0
            spectra = []
            for _spectra, idx_start, idx_end in iter_spectra(return_index=True):
                _indexes_to_serialize = [i - idx_start for i in indexes_to_serialize if idx_start <= i <= idx_end]
                _spectra = list(map(lambda i: _spectra[i], _indexes_to_serialize))

                if count + len(_spectra) >= 1000 * (dump_count + 1):
                    count_to_add = 1000 * (dump_count + 1) - count
                    count += len(_spectra[:count_to_add])
                    spectra += _spectra[:count_to_add]

                    _new_spectra_idx_start = count - 1000
                    _new_spectra_idx_end = count - 1
                    _new_spectra_path = os.path.join(output_dir,
                                                     f'{_new_spectra_idx_start}-{_new_spectra_idx_end}.pickle')

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
            _new_spectra_path = os.path.join(output_dir, f'{_new_spectra_idx_start}-{_new_spectra_idx_end}.pickle')

            with open(_new_spectra_path, 'wb') as f_new:
                pickle.dump(spectra, f_new)
                f_new.flush()


def iter_spectra(return_path=False, return_index=False):
    dir_path = './serialized_spectra'
    path_vs_index_list = []

    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and filename.endswith('.pickle'):
            filename_without_ext = os.path.splitext(filename)[0]
            start_idx = int(filename_without_ext.split('-')[0])
            end_idx = int(filename_without_ext.split('-')[1])
            path_vs_index_list.append((path, start_idx, end_idx))

    path_vs_index_list.sort(key=lambda x: x[1])

    for pickle_path, start_idx, end_idx in path_vs_index_list:
        with open(pickle_path, 'rb') as f:
            spectra = pickle.load(f)

        if return_path and return_index:
            yield spectra, pickle_path, start_idx, end_idx
        elif return_path:
            yield spectra, pickle_path
        elif return_index:
            yield spectra, start_idx, end_idx
        else:
            yield spectra


def iter_filtered_spectra(return_path=False, return_index=False):
    dir_path = './serialized_spectra/filtered'
    path_vs_index_list = []

    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and filename.endswith('.pickle'):
            filename_without_ext = os.path.splitext(filename)[0]
            start_idx = int(filename_without_ext.split('-')[0])
            end_idx = int(filename_without_ext.split('-')[1])
            path_vs_index_list.append((path, start_idx, end_idx))

    path_vs_index_list.sort(key=lambda x: x[1])

    for pickle_path, start_idx, end_idx in path_vs_index_list:
        with open(pickle_path, 'rb') as f:
            spectra = pickle.load(f)

        if return_path and return_index:
            yield spectra, pickle_path, start_idx, end_idx
        elif return_path:
            yield spectra, pickle_path
        elif return_index:
            yield spectra, start_idx, end_idx
        else:
            yield spectra


def get_serialized_spectra_paths(dir_path) -> list:
    """
    Parameters
    ----------
    dir_path : str

    Returns
    -------
    list
        If dir_path includes '0-999.pickle', '1000-1999.pickle', '2000-2110.pickle',
        the following list will be returned.
        
        list[('dir_path/0-999.pickle', 0, 999),
             ('dir_path/1000-1999.pickle', 1000, 1999),
             ('dir_path/2000-2110.pickle', 2000, 2110)]
    """
    path_vs_index_list = []

    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and filename.endswith('.pickle'):
            filename_without_ext = os.path.splitext(filename)[0]
            start_idx = int(filename_without_ext.split('-')[0])
            end_idx = int(filename_without_ext.split('-')[1])
            path_vs_index_list.append((path, start_idx, end_idx))

    path_vs_index_list.sort(key=lambda x: x[1])

    return path_vs_index_list


def get_sample_spectra_paths(dir_path='./serialized_spectra/grouped/sample'):
    """
    Parameters
    ----------
    dir_path : str

    Returns
    -------
    list
        If dir_path includes '0-999.pickle', '1000-1999.pickle', '2000-2110.pickle',
        the following list will be returned.

        list[('dir_path/0-999.pickle', 0, 999),
             ('dir_path/1000-1999.pickle', 1000, 1999),
             ('dir_path/2000-2110.pickle', 2000, 2110)]
    """
    
    return get_serialized_spectra_paths(dir_path)


def get_ref_spectra_paths(parent_dir_path='./serialized_spectra/grouped'):
    """
    Parameters
    ----------
    parent_dir_path : str
        A path of folders which include referene spectra files.

    Returns
    -------
    list
        parent_dir_path
            |- ref_folder_1
                |- 0-999.pickle
                |- 1000-1999.pickle
                |- 2000-2110.pickle
            |- ref_folder_2
                |- 0-700.pickle
        
        For a directory structure like the above, the following dictionary will be returned.
        
        dict{'ref_folder_1': [('dir_path/0-999.pickle', 0, 999),
                              ('dir_path/1000-1999.pickle', 1000, 1999),
                              ('dir_path/2000-2110.pickle', 2000, 2110)]
             'ref_folder_2': [('dir_path/0-700.pickle', 0, 700)]}
    """

    ref_folders = []

    for name in os.listdir(parent_dir_path):
        if name == 'sample':
            continue

        path = os.path.join(parent_dir_path, name)
        if os.path.isdir(path):
            ref_folders.append(path)

    dataset_keyword_vs_spectra_paths = {}
    for ref_folder in ref_folders:
        keyword = os.path.basename(ref_folder)
        paths = get_serialized_spectra_paths(ref_folder)
        dataset_keyword_vs_spectra_paths[keyword] = paths

    return dataset_keyword_vs_spectra_paths


def get_grouped_spectra_dirs(parent_dir_path) -> list:
    """
    Parameters
    ----------
    parent_dir_path : str

    Returns
    -------
    list
        If parent_dir_path includes 'ref_dataset_0', 'ref_dataset_1', 'ref_dataset_2' folders,
        the following list will be returned.

        list[('parent_dir_path/ref_dataset_0', 0),
             ('parent_dir_path/ref_dataset_1', 1),
             ('parent_dir_path/ref_dataset_2', 2)]
    """
    path_vs_index_list = []

    for filename in os.listdir(parent_dir_path):
        path = os.path.join(parent_dir_path, filename)
        if os.path.isdir(path) and re.search(r'\d+', filename):
            matches = re.findall(r'\d+', filename)
            index = int(matches[-1])
            path_vs_index_list.append((path, index))

    path_vs_index_list.sort(key=lambda x: x[1])

    return path_vs_index_list
