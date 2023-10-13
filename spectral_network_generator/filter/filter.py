import h5py
import logging
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import re
from my_parser.spectrum_metadata_parser import get_chunks

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


def remove_blank_spectra_from_sample_spectra(config_obj, mz_tolerance=0.01, rt_in_sec_tolerance=30):
    with h5py.File('./spectrum_metadata.h5', 'a') as h5:
        if not config_obj.list_blank_file_path:
            return

        blank_arr = h5['metadata'][(h5['metadata'].fields('tag')[()].astype(str) == 'blank')]
        blank_arr = blank_arr[(blank_arr['precursor_mz'] != 0) | (blank_arr['rt_in_sec'] != 0)]

        if not blank_arr.size:
            return

        for arr, chunk_start, chunk_end in get_chunks('filtered/metadata'):
            sample_arr = arr[arr['tag'].astype(str) == 'sample']
            ref_idx_arr = arr[arr['tag'].astype(str) == 'ref']['index']

            if sample_arr.size:
                mask = np.array([True] * sample_arr.shape[0])
                for (_blank_mz, _blank_rt) in blank_arr[['precursor_mz', 'rt_in_sec']]:
                    _mask_mz_is_not_zero = sample_arr['precursor_mz'] != 0
                    _mask_rt_is_not_zero = sample_arr['rt_in_sec'] != 0
                    _mask_mz_delta_is_less_than_tol = abs(sample_arr['precursor_mz'] - _blank_mz) < mz_tolerance
                    _mask_rt_delta_is_less_than_tol = abs(sample_arr['rt_in_sec'] - _blank_rt) < rt_in_sec_tolerance
                    _mask = (_mask_mz_is_not_zero & _mask_rt_is_not_zero
                             & _mask_mz_delta_is_less_than_tol & _mask_rt_delta_is_less_than_tol)
                    mask = np.logical_and(np.invert(_mask), mask)

                filtered_sample_idx_arr = sample_arr[mask]['index']
                idx_arr = np.append(filtered_sample_idx_arr, ref_idx_arr)
            else:
                idx_arr = ref_idx_arr

            idx_arr.sort()
            arr_to_retain = arr[np.isin(arr['index'], idx_arr)]
            if '_metadata' not in h5['filtered'].keys():
                h5.create_dataset(f'filtered/_metadata', data=arr_to_retain,
                                  shape=arr_to_retain.shape, maxshape=(None,))
            else:
                h5['filtered/_metadata'].resize(h5['filtered/_metadata'].len() + arr_to_retain.shape[0], axis=0)
                h5['filtered/_metadata'][-arr_to_retain.shape[0]:] = arr_to_retain
            h5.flush()

        h5['filtered/metadata'].resize(h5['filtered/_metadata'].len(), axis=0)
        h5['filtered/metadata'][:] = h5['filtered/_metadata'][:]
        del h5['filtered/_metadata']
        h5.flush()


def filter_reference_spectra(config_obj, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    LOGGER.info('Filter reference spectra')

    def _avoid_filter(a):
        if a in config_obj.list_filename_avoid_filter:
            return True
        else:
            return False

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
            sample_idx_arr = arr[arr['tag'] == b'sample']['index']

            if not ref_arr.size:
                idx_arr = sample_idx_arr
            else:
                _mask_all_true = np.array([True] * ref_arr.shape[0])
                mask_avoid_filter = _mask_all_true

                # whether to filter ---------------------------------------------------
                if config_obj.list_filename_avoid_filter:
                    _filter = np.frompyfunc(_avoid_filter, 1, 1)
                    mask_avoid_filter = np.isin(np.char.decode(ref_arr['source_filename'].astype(np.bytes_), encoding='utf8'),
                                                config_obj.list_filename_avoid_filter)

                unfiltered_ref_idx_arr = ref_arr[mask_avoid_filter]['index']
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
                    if config_obj.list_filename_compound_dat_for_filter:
                        mask_compound = np.isin(ref_arr['inchikey'].astype(str), metacyc_inchikeys)

                    mask_all = (mask_author
                                & mask_name
                                & mask_instrument_type
                                & mask_ionization_mode
                                & mask_precursor_type
                                & mask_fragmentation_type
                                & mask_min_number_of_peaks
                                & mask_precursor_mass_exists
                                & mask_compound).astype(bool)

                    filtered_ref_idx_arr = ref_arr[mask_all]['index']
                    idx_arr = np.append(sample_idx_arr, filtered_ref_idx_arr)
                    idx_arr = np.append(idx_arr, unfiltered_ref_idx_arr)

            idx_arr.sort()
            arr_to_retain = arr[np.isin(arr['index'], idx_arr)]
            if '_metadata' not in h5['filtered'].keys():
                h5.create_dataset(f'filtered/_metadata', data=arr_to_retain,
                                  shape=arr_to_retain.shape, maxshape=(None,))
            else:
                h5['filtered/_metadata'].resize(h5['filtered/_metadata'].len() + arr_to_retain.shape[0], axis=0)
                h5['filtered/_metadata'][-arr_to_retain.shape[0]:] = arr_to_retain
            h5.flush()

        h5['filtered/metadata'].resize(h5['filtered/_metadata'].len(), axis=0)
        h5['filtered/metadata'][:] = h5['filtered/_metadata'][:]
        del h5['filtered/_metadata']
        h5.flush()
