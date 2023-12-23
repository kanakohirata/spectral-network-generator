import h5py
import logging
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import os
import re

import pandas as pd

from my_parser.spectrum_metadata_parser import get_chunks
from utils import split_array

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


def remove_blank_spectra_from_sample_spectra_old(mz_tolerance=0.01, rt_tolerance=0.1):
    with h5py.File('./spectrum_metadata.h5', 'a') as h5:
        blank_arr = h5['metadata'][(h5['metadata'].fields('tag')[()].astype(str) == 'blank')
                                   & (h5['metadata'].fields('precursor_mz')[()] != 0)
                                   & (h5['metadata'].fields('rt_in_sec')[()] != 0)]
        
        if not blank_arr.size:
            return
        blank_arr = blank_arr[['precursor_mz', 'rt_in_sec']]

        for arr, chunk_start, chunk_end in get_chunks('filtered/metadata'):
            sample_arr = arr[(arr['tag'].astype(str) == 'sample')
                             & (arr['precursor_mz'] != 0)
                             & (arr['rt_in_sec'] != 0)]
            non_target_idx_arr = np.setdiff1d(arr['index'], sample_arr['index'])

            if sample_arr.size:
                sample_mz = sample_arr['precursor_mz']
                sample_rt = sample_arr['rt_in_sec']

                mz_diff = np.abs(sample_mz[:, np.newaxis] - blank_arr['precursor_mz']) <= mz_tolerance
                rt_diff = np.abs(sample_rt[:, np.newaxis] - blank_arr['rt_in_sec']) <= rt_tolerance

                mz_and_rt_diff = np.logical_and(mz_diff, rt_diff)
                mz_and_rt_diff_inverted = np.invert(mz_and_rt_diff)
                mask = np.all(mz_and_rt_diff_inverted, axis=1)

                filtered_sample_idx_arr = sample_arr['index'][mask]
                idx_arr = np.append(filtered_sample_idx_arr, non_target_idx_arr)
            else:
                idx_arr = non_target_idx_arr

            idx_arr.sort()
            arr_to_retain = arr[np.isin(arr['index'], idx_arr)]
            if '_metadata' not in h5['filtered'].keys():
                h5.create_dataset(f'filtered/_metadata', data=arr_to_retain,
                                  shape=arr_to_retain.shape, maxshape=(None,))
            else:
                h5['filtered/_metadata'].resize(h5['filtered/_metadata'].len() + arr_to_retain.shape[0], axis=0)
                h5['filtered/_metadata'][-arr_to_retain.shape[0]:] = arr_to_retain
            h5.flush()

        del h5['filtered/metadata']
        h5.flush()
        
        h5.create_dataset('filtered/metadata', data=h5['filtered/_metadata'][()], shape=h5['filtered/_metadata'].shape, maxshape=(None,))
        del h5['filtered/_metadata']
        h5.flush()


def remove_blank_spectra_from_sample_spectra(blank_metadata_path, sample_metadata_path, output_path,
                                             mz_tolerance=0.01, rt_tolerance=0.1, export_tsv=False):
    if not os.path.isfile(blank_metadata_path):
        return

    # Load blank metadata array
    blank_arr = np.load(blank_metadata_path, allow_pickle=True)
    if not blank_arr.size:
        return

    # Extract blank array where 'precursor_mz' and 'rt_in_sec' are not 0.
    blank_arr = blank_arr[['precursor_mz', 'rt_in_sec']]
    blank_arr = blank_arr[(blank_arr['precursor_mz'] != 0) & (blank_arr['rt_in_sec'] != 0)]

    if not blank_arr.size:
        return

    # Load sample metadata array
    sample_arr_all = np.load(sample_metadata_path, allow_pickle=True)
    if not sample_arr_all.size:
        return

    # Make output folder if it does not exist.
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # If sample_metadata_path == output_path,
    # sample_metadata_path will be updated after outputting to the temp_output_path file.
    temp_output_path = os.path.splitext(output_path)[0] + '__temp.npy'

    arr_to_retain = None

    # Iterate sample metadata array per 10000 rows.
    for arr, chunk_start, chunk_end in split_array(sample_arr_all, 10000):
        # Extract sample array where 'precursor_mz' and 'rt_in_sec' are not 0.
        mask_target_sample = (arr['precursor_mz'] != 0) & (arr['rt_in_sec'] != 0)
        sample_arr = arr[mask_target_sample]
        idx_arr_to_retain = arr['index'][~mask_target_sample]

        if sample_arr.size:
            sample_mz = sample_arr['precursor_mz']
            sample_rt = sample_arr['rt_in_sec']

            mz_diff = np.abs(sample_mz[:, np.newaxis] - blank_arr['precursor_mz']) <= mz_tolerance
            rt_diff = np.abs(sample_rt[:, np.newaxis] - blank_arr['rt_in_sec']) <= rt_tolerance

            mz_and_rt_diff = np.logical_and(mz_diff, rt_diff)
            mz_and_rt_diff_inverted = np.invert(mz_and_rt_diff)
            mask = np.all(mz_and_rt_diff_inverted, axis=1)

            filtered_sample_idx_arr = sample_arr['index'][mask]
            idx_arr = np.append(filtered_sample_idx_arr, idx_arr_to_retain)
        else:
            idx_arr = idx_arr_to_retain

        if idx_arr.size:
            idx_arr.sort()
            arr_to_retain = arr[np.isin(arr['index'], idx_arr)]

            # Append arr_to_retain to an already existing array.
            if os.path.isfile(temp_output_path):
                existing_arr = np.load(temp_output_path, allow_pickle=True)
                arr_to_retain = np.hstack((existing_arr, arr_to_retain))

            with open(temp_output_path, 'wb') as f:
                np.save(f, arr_to_retain)
                f.flush()

    if arr_to_retain is not None:
        with open(output_path, 'wb') as f:
            np.save(f, arr_to_retain)
            f.flush()
            
        if export_tsv:
            tsv_path = os.path.splitext(output_path)[0] + '.tsv'
            df = pd.DataFrame.from_records(arr_to_retain)
            df.to_csv(tsv_path, sep='\t', index=False)

        # Remove temp_output_path file
        os.remove(temp_output_path)


def remove_sample_spectra_with_no_precursor_mz(sample_metadata_path, output_path, export_tsv=False):
    # Load sample metadata array
    sample_arr_all = np.load(sample_metadata_path, allow_pickle=True)
    if not sample_arr_all.size:
        return

    # Make output folder if it does not exist.
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # If sample_metadata_path == output_path,
    # sample_metadata_path will be updated after outputting to the temp_output_path file.
    temp_output_path = os.path.splitext(output_path)[0] + '__temp.npy'

    arr_to_retain = None

    # Iterate sample metadata array per 10000 rows.
    for arr, _, _ in split_array(sample_arr_all, 10000):
        mask = arr['precursor_mz'] > 0

        if not np.any(mask):
            continue

        arr_to_retain = arr[mask]

        # Append arr_to_retain to an already existing array.
        if os.path.isfile(temp_output_path):
            existing_arr = np.load(temp_output_path, allow_pickle=True)
            arr_to_retain = np.hstack((existing_arr, arr_to_retain))

        with open(temp_output_path, 'wb') as f:
            np.save(f, arr_to_retain)
            f.flush()

    if arr_to_retain is not None:
        with open(output_path, 'wb') as f:
            np.save(f, arr_to_retain)
            f.flush()

        if export_tsv:
            tsv_path = os.path.splitext(output_path)[0] + '.tsv'
            df = pd.DataFrame.from_records(arr_to_retain)
            df.to_csv(tsv_path, sep='\t', index=False)

        # Remove temp_output_path file
        os.remove(temp_output_path)


def filter_sample_spectra(config_obj, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    with h5py.File('./spectrum_metadata.h5', 'a') as h5:
        for arr, chunk_start, chunk_end in get_chunks('filtered/metadata'):
            ref_idx_arr = arr['index'][arr['tag'] == b'ref']
            sample_arr = arr[arr['tag'] == b'sample']

            if not sample_arr.size:
                idx_arr = ref_idx_arr
            else:
                _mask_all_true = np.array([True] * sample_arr.shape[0])

                # whether a record has precursor mass or not --------------------------------------------------
                mask_precursor_mass_exists = _mask_all_true
                if config_obj.remove_spec_wo_prec_mz:
                    mask_precursor_mass_exists = sample_arr['precursor_mz'] > 0

                mask_all = mask_precursor_mass_exists

                filtered_sample_idx_arr = sample_arr['index'][mask_all]
                idx_arr = np.append(filtered_sample_idx_arr, ref_idx_arr)

            if idx_arr.size:
                idx_arr.sort()
                arr_to_retain = arr[np.isin(arr['index'], idx_arr)]
                if '_metadata' not in h5['filtered'].keys():
                    h5.create_dataset(f'filtered/_metadata', data=arr_to_retain,
                                    shape=arr_to_retain.shape, maxshape=(None,))
                else:
                    h5['filtered/_metadata'].resize(h5['filtered/_metadata'].len() + arr_to_retain.shape[0], axis=0)
                    h5['filtered/_metadata'][-arr_to_retain.shape[0]:] = arr_to_retain
                h5.flush()

        del h5['filtered/metadata']
        h5.flush()
        
        h5.create_dataset('filtered/metadata', data=h5['filtered/_metadata'][()], shape=h5['filtered/_metadata'].shape, maxshape=(None,))
        del h5['filtered/_metadata']
        h5.flush()


def filter_reference_spectra_old(config_obj, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    LOGGER.info('Filter reference spectra')

    def _author_filter(a):
        result = False
        for _author_keyword in config_obj.list_author:
            if not _author_keyword:
                continue
            if re.search(_author_keyword, a, flags=re.IGNORECASE):
                result = True
            else:
                result = result or False

        return result

    def _key_character_filter(a):
        result = True
        for _key_characters in config_obj.list_name_key_characters_to_remove:
            if not _key_characters:
                continue
            if re.search(_key_characters, a, flags=re.IGNORECASE):
                result = False
            else:
                result = result and True

        return result

    def _instrument_type_filter(a):
        list_pattern_QqQ = ["QqQ", "QQQ", "Q3", "qqq", "Quattro_QQQ", "QQQ/MS"]
        list_pattern_TOF = ["LC-ESI-QTOF", "ESI-QTOF", "Q-TOF", "TOF", "tof", "ToF", "qTof",
                            "Flow-injection QqQ/MS", "LC-Q-TOF/MS"]
        list_pattern_FT = ["FT", "ESI-QFT", "LC-ESI-QFT", "LC-ESI-ITFT", "ESI-ITFT", "orbitrap",
                           "Orbitrap",
                           "Q-Exactive Plus", "Q-Exactive"]
        list_pattern_GC = ["GC-EI-Q", "GC-EI-QQ", "GC-EI-TOF", "EI-B", "GC-MS", "GC-MS-EI"]

        if config_obj.instrument_type in ('LC-ESI-ITFT', 'LC-ESI-QFT'):
            if a == config_obj.instrument_type:
                return True

        elif config_obj.instrument_type == 'FT':
            if a in list_pattern_FT:
                return True

        elif config_obj.instrument_type == 'TOF':
            if a in list_pattern_TOF:
                return True

        elif config_obj.instrument_type == 'TOF_FT':
            if a in list_pattern_TOF or a in list_pattern_FT:  # TODO: Check
                return True

        elif config_obj.instrument_type == 'QqQ':
            if a in list_pattern_QqQ:
                return True

        elif config_obj.instrument_type == 'GC':
            if a in list_pattern_GC:
                return True

        return False

    def _ionization_filter(a, b):
        if (a == config_obj.ionization
                or re.search(f'({config_obj.ionization})', a, flags=re.IGNORECASE)
                or re.search(config_obj.ionization, b, flags=re.IGNORECASE)):
            return True
        else:
            return False

    def _fragmentation_type_filter(a):
        list_pattern_frag = []
        if config_obj.fragmentation_type == "CID":
            list_pattern_frag = ["CID", "LOW-ENERGY CID"]
        elif config_obj.fragmentation_type == "HCD":
            list_pattern_frag = ["HCD", "HIGH-ENERGY CID"]

        result = False
        for _p in list_pattern_frag:
            if re.search(_p, a, flags=re.IGNORECASE):
                result = True
            else:
                result = result or False

        return result

    with h5py.File('./metacyc_for_filter.h5', 'r') as metacyc_h5:
        metacyc_dset = metacyc_h5['compound']
        if metacyc_dset.size:
            metacyc_inchikeys = metacyc_dset.fields('inchikey')[()].astype(str)
        else:
            metacyc_inchikeys = np.array([]).astype(str)


    with h5py.File('./spectrum_metadata.h5', 'a') as h5:
        for arr, chunk_start, chunk_end in get_chunks('filtered/metadata'):
            ref_arr = arr[arr['tag'] == b'ref']
            sample_idx_arr = arr['index'][arr['tag'] == b'sample']

            if not ref_arr.size:
                idx_arr = sample_idx_arr
            else:
                _mask_all_true = np.array([True] * ref_arr.shape[0])
                mask_avoid_filter = np.invert(_mask_all_true)

                # whether to filter ---------------------------------------------------
                if config_obj.list_filename_avoid_filter:
                    mask_avoid_filter = np.isin(np.char.decode(ref_arr['source_filename'].astype(np.bytes_), encoding='utf8'),
                                                config_obj.list_filename_avoid_filter)

                unfiltered_ref_idx_arr = ref_arr['index'][mask_avoid_filter]
                ref_arr = ref_arr[np.invert(mask_avoid_filter)]

                if not ref_arr.size:
                    idx_arr = np.append(sample_idx_arr, unfiltered_ref_idx_arr)
                else:
                    _mask_all_true = np.array([True] * ref_arr.shape[0])

                    # author ---------------------------------------------------
                    mask_author = _mask_all_true
                    if config_obj.list_author:
                        _filter = np.frompyfunc(_author_filter, 1, 1)
                        mask_author = _filter(np.char.decode(ref_arr['author'].astype(np.bytes_), encoding='utf8'))

                    # name ---------------------------------------------------
                    # basically to remove PCs
                    mask_name = _mask_all_true
                    if config_obj.list_name_key_characters_to_remove:
                        logger.debug(config_obj.list_name_key_characters_to_remove)
                        _filter = np.frompyfunc(_key_character_filter, 1, 1)
                        mask_name = _filter(np.char.decode(ref_arr['compound_name'].astype(np.bytes_), encoding='utf8'))

                    # instrument type ---------------------------------------------------
                    mask_instrument_type = _mask_all_true
                    if config_obj.instrument_type:
                        _filter = np.frompyfunc(_instrument_type_filter, 1, 1)
                        mask_instrument_type = _filter(ref_arr['instrument_type'].astype(str))

                    # ionization mode ---------------------------------------------------
                    mask_ionization_mode = _mask_all_true
                    if config_obj.ionization:
                        # if ionization mode is specified in config....   probably ESI
                        _filter = np.frompyfunc(_ionization_filter, 2, 1)
                        mask_ionization_mode = _filter(ref_arr['ionization_mode'].astype(str),
                                                       ref_arr['instrument_type'].astype(str))

                        _ionization_mode_ref_arr = np.where(mask_ionization_mode,
                                                            config_obj.ionization,
                                                            ref_arr['ionization_mode'].astype(str))

                        ref_arr['ionization_mode'] = _ionization_mode_ref_arr

                    # precursor ion type ---------------------------------------------------
                    # [M+H]+ , [M-H]-  etc.
                    mask_precursor_type = _mask_all_true
                    if config_obj.list_precursor_type:
                        mask_precursor_type = np.isin(ref_arr['precursor_type'].astype(str), config_obj.list_precursor_type)

                    # fragmentation type ---------------------------------------------------
                    # you should detect like "low energy CID" with CID key pattern
                    mask_fragmentation_type = _mask_all_true
                    if config_obj.fragmentation_type:
                        _filter = np.frompyfunc(_fragmentation_type_filter, 1, 1)
                        mask_fragmentation_type = _filter(ref_arr['fragmentation_type'].astype(str))

                    # minimum number of peaks ---------------------------------------------------
                    mask_min_number_of_peaks = _mask_all_true
                    if config_obj.min_number_of_peaks > 0:
                        mask_min_number_of_peaks = ref_arr['number_of_peaks'] > config_obj.min_number_of_peaks

                    # whether a record has precursor mass or not --------------------------------------------------
                    mask_precursor_mass_exists = _mask_all_true
                    if config_obj.remove_spec_wo_prec_mz:
                        mask_precursor_mass_exists = ref_arr['precursor_mz'] > 0

                    # compound filter --------------------------------------------------
                    mask_compound = _mask_all_true
                    if config_obj.list_path_compound_dat_for_filter and metacyc_inchikeys.size:
                        mask_compound = np.isin(ref_arr['inchikey'].astype(str), metacyc_inchikeys)

                    # keyword filter --------------------------------------------------
                    if config_obj.ref_split_category == 'cmpd_classification_superclass':
                        keyword_arr = ref_arr['cmpd_classification_superclass_list']
                    elif config_obj.ref_split_category == 'cmpd_classification_class':
                        keyword_arr = ref_arr['cmpd_classification_class_list']
                    elif config_obj.ref_split_category == 'cmpd_pathway':
                        keyword_arr = ref_arr['pathway_unique_id_list']
                    else:
                        keyword_arr = np.empty()
                    
                    # to select --------------------------------------------------
                    mask_select_keyword = _mask_all_true
                    if config_obj.list_ref_select_keyword and keyword_arr.size:
                        mask_select_keyword = np.isin(keyword_arr, config_obj.list_ref_select_keyword)
                    
                    # to exclude --------------------------------------------------
                    mask_exclude_keyword = _mask_all_true
                    if config_obj.list_ref_exclude_keyword and keyword_arr.size:
                        mask_exclude_keyword = np.invert(np.isin(keyword_arr, config_obj.list_ref_exclude_keyword))

                    # external compound file filter -------------------------------
                    mask_external_compound_file = _mask_all_true
                    if config_obj.external_file_filter_mode:
                        mask_external_compound_file = ref_arr['pathway_common_name_list'] != b''
                    
                    
                    mask_all = (mask_author
                                & mask_name
                                & mask_instrument_type
                                & mask_ionization_mode
                                & mask_precursor_type
                                & mask_fragmentation_type
                                & mask_min_number_of_peaks
                                & mask_precursor_mass_exists
                                & mask_compound
                                & mask_select_keyword
                                & mask_exclude_keyword
                                & mask_external_compound_file).astype(bool)

                    filtered_ref_idx_arr = ref_arr['index'][mask_all]
                    idx_arr = np.append(sample_idx_arr, filtered_ref_idx_arr)
                    idx_arr = np.append(idx_arr, unfiltered_ref_idx_arr)
            
            if idx_arr.size:
                idx_arr.sort()
                arr_to_retain = arr[np.isin(arr['index'], idx_arr)]
                if '_metadata' not in h5['filtered'].keys():
                    h5.create_dataset(f'filtered/_metadata', data=arr_to_retain,
                                      shape=arr_to_retain.shape, maxshape=(None,))
                else:
                    h5['filtered/_metadata'].resize(h5['filtered/_metadata'].len() + arr_to_retain.shape[0], axis=0)
                    h5['filtered/_metadata'][-arr_to_retain.shape[0]:] = arr_to_retain
                h5.flush()

        del h5['filtered/metadata']
        h5.flush()
        
        h5.create_dataset('filtered/metadata', data=h5['filtered/_metadata'][()], shape=h5['filtered/_metadata'].shape, maxshape=(None,))
        del h5['filtered/_metadata']
        h5.flush()


def filter_reference_spectra(config_obj, ref_metadata_path, output_path, metacyc_compound_path,
                             export_tsv=False, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    LOGGER.info('Filter reference spectra')

    # filtering functions ----------------------------------------------------------------
    def _author_filter(a):
        result = False
        for _author_keyword in config_obj.list_author:
            if not _author_keyword:
                continue
            if re.search(_author_keyword, a, flags=re.IGNORECASE):
                result = True
            else:
                result = result or False

        return result

    def _key_character_filter(a):
        result = True
        for _key_characters in config_obj.list_name_key_characters_to_remove:
            if not _key_characters:
                continue
            if _key_characters in a:
                result = False
            else:
                result = result and True

        return result

    def _instrument_type_filter(a):
        list_pattern_QqQ = ["QqQ", "QQQ", "Q3", "qqq", "Quattro_QQQ", "QQQ/MS"]
        list_pattern_TOF = ["LC-ESI-QTOF", "ESI-QTOF", "Q-TOF", "TOF", "tof", "ToF", "qTof",
                            "Flow-injection QqQ/MS", "LC-Q-TOF/MS"]
        list_pattern_FT = ["FT", "ESI-QFT", "LC-ESI-QFT", "LC-ESI-ITFT", "ESI-ITFT", "orbitrap",
                           "Orbitrap",
                           "Q-Exactive Plus", "Q-Exactive"]
        list_pattern_GC = ["GC-EI-Q", "GC-EI-QQ", "GC-EI-TOF", "EI-B", "GC-MS", "GC-MS-EI"]

        if config_obj.instrument_type == 'FT':
            if a in list_pattern_FT:
                return True

        elif config_obj.instrument_type == 'TOF':
            if a in list_pattern_TOF:
                return True

        elif config_obj.instrument_type == 'TOF_FT':
            if a in list_pattern_TOF or a in list_pattern_FT:  # TODO: Check
                return True

        elif config_obj.instrument_type == 'QqQ':
            if a in list_pattern_QqQ:
                return True

        elif config_obj.instrument_type == 'GC':
            if a in list_pattern_GC:
                return True

        else:
            if a == config_obj.instrument_type:
                return True

        return False

    def _fragmentation_type_filter(a):
        list_pattern_frag = []
        if config_obj.fragmentation_type == "CID":
            list_pattern_frag = ["CID", "LOW-ENERGY CID"]
        elif config_obj.fragmentation_type == "HCD":
            list_pattern_frag = ["HCD", "HIGH-ENERGY CID"]

        result = False
        for _p in list_pattern_frag:
            if re.search(_p, a, flags=re.IGNORECASE):
                result = True
            else:
                result = result or False

        return result
    # ------------------------------------------------------------------------------------

    # Load inchikeys in metacyc file.
    metacyc_inchikeys = np.array([]).astype(str)
    if os.path.isfile(metacyc_compound_path):
        metacyc_arr = np.load(metacyc_compound_path, allow_pickle=True)
        if metacyc_arr.size:
            metacyc_inchikeys = metacyc_arr['inchikey']

    # If ref_metadata_path == output_path,
    # ref_metadata_path will be updated after outputting to the temp_output_path file.
    temp_output_path = os.path.splitext(output_path)[0] + '__temp.npy'

    arr_all = np.load(ref_metadata_path, allow_pickle=True)
    arr_to_retain = None
    for arr, chunk_start, chunk_end in split_array(arr_all, 10000):
        _mask_all_true = np.array([True] * arr.shape[0])
        mask_avoid_filter = np.invert(_mask_all_true)

        # whether to filter ---------------------------------------------------
        if config_obj.list_filename_avoid_filter:
            _a = arr['source_filename']
            mask_avoid_filter = np.isin(arr['source_filename'], config_obj.list_filename_avoid_filter)

        unfiltered_ref_idx_arr = arr['index'][mask_avoid_filter]
        ref_arr = arr[np.invert(mask_avoid_filter)]

        if not ref_arr.size:
            idx_arr = unfiltered_ref_idx_arr
        else:
            _mask_all_true = np.array([True] * ref_arr.shape[0])

            # author ---------------------------------------------------
            mask_author = _mask_all_true
            if config_obj.list_author:
                _filter = np.frompyfunc(_author_filter, 1, 1)
                mask_author = _filter(ref_arr['author'])
                if not np.any(mask_author):
                    logger.warning('!!!')

            # name ---------------------------------------------------
            # basically to remove PCs
            mask_name = _mask_all_true
            if config_obj.list_name_key_characters_to_remove:
                logger.debug(config_obj.list_name_key_characters_to_remove)
                _filter = np.frompyfunc(_key_character_filter, 1, 1)
                mask_name = _filter(ref_arr['compound_name'])
                if not np.any(mask_name):
                    logger.warning('!!!')

            # instrument type ---------------------------------------------------
            mask_instrument_type = _mask_all_true
            if config_obj.instrument_type:
                _filter = np.frompyfunc(_instrument_type_filter, 1, 1)
                mask_instrument_type = _filter(ref_arr['instrument_type'])
                if not np.any(mask_instrument_type):
                    logger.warning('!!!')

            # ionization mode ---------------------------------------------------
            mask_ionization_mode = _mask_all_true
            if config_obj.ionization:
                mask_ionization_mode = ref_arr['ionization_mode'] == config_obj.ionization
                if not np.any(mask_ionization_mode):
                    logger.warning('!!!')

            # ion mode ---------------------------------------------------------
            mask_ion_mode = _mask_all_true
            if config_obj.ion_mode:
                mask_ion_mode = ref_arr['ion_mode'] == config_obj.ion_mode
                if not np.any(mask_ion_mode):
                    logger.warning('!!!')

            # precursor ion type ---------------------------------------------------
            # [M+H]+ , [M-H]-  etc.
            mask_precursor_type = _mask_all_true
            if config_obj.list_precursor_type:
                mask_precursor_type = np.isin(ref_arr['precursor_type'], config_obj.list_precursor_type)
                if not np.any(mask_precursor_type):
                    logger.warning('!!!')

            # fragmentation type ---------------------------------------------------
            # you should detect like "low energy CID" with CID key pattern
            mask_fragmentation_type = _mask_all_true
            if config_obj.fragmentation_type:
                _filter = np.frompyfunc(_fragmentation_type_filter, 1, 1)
                mask_fragmentation_type = _filter(ref_arr['fragmentation_type'])
                if not np.any(mask_fragmentation_type):
                    logger.warning('!!!')

            # minimum number of peaks ---------------------------------------------------
            mask_min_number_of_peaks = _mask_all_true
            if config_obj.min_number_of_peaks > 0:
                mask_min_number_of_peaks = ref_arr['number_of_peaks'] >= config_obj.min_number_of_peaks
                if not np.any(mask_min_number_of_peaks):
                    logger.warning('!!!')

            # whether a record has precursor mass or not --------------------------------------------------
            mask_precursor_mass_exists = _mask_all_true
            if config_obj.remove_spec_wo_prec_mz:
                mask_precursor_mass_exists = ref_arr['precursor_mz'] > 0
                if not np.any(mask_precursor_mass_exists):
                    logger.warning('!!!')

            # compound filter --------------------------------------------------
            mask_compound = _mask_all_true
            if config_obj.list_path_compound_dat_for_filter and metacyc_inchikeys.size:
                mask_compound = np.isin(ref_arr['inchikey'], metacyc_inchikeys)
                if not np.any(mask_compound):
                    logger.warning('!!!')

            # keyword filter --------------------------------------------------
            if config_obj.ref_split_category == 'cmpd_classification_superclass':
                keyword_arr = ref_arr['cmpd_classification_superclass']
            elif config_obj.ref_split_category == 'cmpd_classification_class':
                keyword_arr = ref_arr['cmpd_classification_class']
            elif config_obj.ref_split_category == 'cmpd_pathway':
                keyword_arr = ref_arr['pathway_unique_id_list']
            else:
                keyword_arr = np.empty()

            # to select --------------------------------------------------
            mask_select_keyword = _mask_all_true
            if config_obj.list_ref_select_keyword and keyword_arr.size:
                mask_select_keyword = np.isin(keyword_arr, config_obj.list_ref_select_keyword)
                if not np.any(mask_select_keyword):
                    logger.warning('!!!')

            # to exclude --------------------------------------------------
            mask_exclude_keyword = _mask_all_true
            if config_obj.list_ref_exclude_keyword and keyword_arr.size:
                mask_exclude_keyword = np.invert(np.isin(keyword_arr, config_obj.list_ref_exclude_keyword))
                if not np.any(mask_exclude_keyword):
                    logger.warning('!!!')

            # external compound file filter -------------------------------
            mask_external_compound_file = _mask_all_true
            if config_obj.external_file_filter_mode:
                mask_external_compound_file = ref_arr['pathway_common_name_list'] != []
                if not np.any(mask_external_compound_file):
                    logger.warning('!!!')

            mask_all = (mask_author
                        & mask_name
                        & mask_instrument_type
                        & mask_ionization_mode
                        & mask_ion_mode
                        & mask_precursor_type
                        & mask_fragmentation_type
                        & mask_min_number_of_peaks
                        & mask_precursor_mass_exists
                        & mask_compound
                        & mask_select_keyword
                        & mask_exclude_keyword
                        & mask_external_compound_file).astype(bool)

            filtered_ref_idx_arr = ref_arr['index'][mask_all]
            idx_arr = np.append(filtered_ref_idx_arr, unfiltered_ref_idx_arr)

        if idx_arr.size:
            idx_arr.sort()
            arr_to_retain = arr[np.isin(arr['index'], idx_arr)]

            # Append arr_to_retain to an already existing array.
            if os.path.isfile(temp_output_path):
                existing_arr = np.load(temp_output_path, allow_pickle=True)
                arr_to_retain = np.hstack((existing_arr, arr_to_retain))

            with open(temp_output_path, 'wb') as f:
                np.save(f, arr_to_retain)
                f.flush()

    # Write content of temp_output_path to output_path
    if arr_to_retain is not None:
        with open(output_path, 'wb') as f:
            np.save(f, arr_to_retain)
            f.flush()

        if export_tsv:
            tsv_path = os.path.splitext(output_path)[0] + '.tsv'
            df = pd.DataFrame.from_records(arr_to_retain)
            df.to_csv(tsv_path, sep='\t', index=False)

        # Remove temp_output_path file
        os.remove(temp_output_path)


def extract_top_x_peak_rich_old(filename, num_top_x_peak_rich, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    if num_top_x_peak_rich <= 0:
        return
    
    logger.info(f'Extract top {num_top_x_peak_rich} peak rich spectra of {filename}')
    
    with h5py.File('./spectrum_metadata.h5', 'a') as h5:
        dset = h5['filtered/metadata']
        idx_arr = dset.fields('index')[()]
        filename_arr = dset.fields('source_filename')[()].astype(str)

        target_arr = dset[filename_arr == filename]
        if target_arr.shape[0] <= num_top_x_peak_rich:
            return
        target_arr = target_arr[['index', 'number_of_peaks']]
        target_idx_arr = target_arr['index']
        idx_arr_to_retain = np.setdiff1d(idx_arr, target_idx_arr)

        sorted_indices_desc = np.argsort(target_arr['number_of_peaks'])[::-1]
        target_arr = target_arr[sorted_indices_desc][:num_top_x_peak_rich]

        idx_arr_to_retain = np.append(idx_arr_to_retain, target_arr['index'])
        idx_arr_to_retain.sort()
        arr_to_retain = dset[np.isin(idx_arr, idx_arr_to_retain)]

        del h5['filtered/metadata']
        h5.flush()

        h5.create_dataset('filtered/metadata', data=arr_to_retain, shape=arr_to_retain.shape, maxshape=(None,))
        h5.flush()


def extract_top_x_peak_rich(metadata_path, output_path, filename, num_top_x_peak_rich, export_tsv=False, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    if num_top_x_peak_rich <= 0:
        return

    # Make output folder if it does not exist.
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    logger.info(f'Extract top {num_top_x_peak_rich} peak rich spectra of {filename}')

    arr = np.load(metadata_path, allow_pickle=True)

    idx_arr = arr['index']
    filename_arr = arr['source_filename']

    target_arr = arr[filename_arr == filename]
    if target_arr.shape[0] <= num_top_x_peak_rich:
        return
    target_arr = target_arr[['index', 'number_of_peaks']]
    target_idx_arr = target_arr['index']
    idx_arr_to_retain = np.setdiff1d(idx_arr, target_idx_arr)

    sorted_indices_desc = np.argsort(target_arr['number_of_peaks'])[::-1]
    target_arr = target_arr[sorted_indices_desc][:num_top_x_peak_rich]

    idx_arr_to_retain = np.append(idx_arr_to_retain, target_arr['index'])
    idx_arr_to_retain.sort()
    arr_to_retain = arr[np.isin(idx_arr, idx_arr_to_retain)]

    with open(output_path, 'wb') as f:
        np.save(f, arr_to_retain)
        f.flush()

    if export_tsv:
        tsv_path = os.path.splitext(output_path)[0] + '.tsv'
        df = pd.DataFrame.from_records(arr_to_retain)
        df.to_csv(tsv_path, sep='\t', index=False)

