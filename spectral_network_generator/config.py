import configparser
from glob import glob
from copy import deepcopy
import errno
import itertools
import os
import shutil

import pandas as pd


class SpecNetGenConfig:
    def __init__(self):
        self.id = -1
        # [input] section
        self.input_sample_folder_path = ''
        self.input_ref_folder_path = ''
        self.input_blank_folder_path = ''

        # file paths and filenames of inputs will be set automatically.
        self.list_sample_file_path = []
        self.list_ref_file_path = []
        self.list_blank_file_path = []
        self.list_sample_filename = []
        self.list_ref_filename = []
        self.list_blank_filename = []

        self.list_ref_spec_filename = []
        self.list_ref_spec_path = []
        self.ref_split_category = ''
        self.list_ref_select_keyword = []
        self.list_ref_exclude_keyword = []

        self.create_edge_within_layer_ref = 0
        self.list_decoy = []

        self.list_author = ''
        self.list_name_key_characters_to_remove = []
        self.instrument_type = ''
        self.list_precursor_type = []
        self.fragmentation_type = ''
        self.min_number_of_peaks = 1
        self.list_filename_compound_dat_for_filter = []
        self.list_filename_avoid_filter = []

        self.filename_top_X_peak_rich = []
        self.num_top_X_peak_rich = 1000

        # spectral processing
        self.remove_low_intensity_peaks = 0.0005
        self.deisotope_width = 1
        self.deisotope_int_ratio = 1
        self.topN_binned_ranges_topN_number = 10
        self.topN_binned_ranges_bin_size = 10
        self.intensity_convert_mode = 0

        # matching related
        self.spec_matching_mode = 1
        self.mz_tol = 0
        self.matching_top_N_input = 20
        self.match_peaklist_length = 1

        self.score_threshold_to_output = 0
        self.minimum_peak_match_to_output = 0
        self.min_num_input_peaks_for_valid_bonanza = 0

        self.output_filename = ''
        self.output_folder_path = ''
        self.fo_score_vs_chemsim = ''

        self.class_matching_correlation = 0

        self.flag_write_log = 0
        self.flag_display_process = 0

        self.filename_metacyc_cmpd_dat = ''
        self.filename_metacyc_pathway_dat = ''
        self.compound_table_folder_path = ''
        self.compound_table_paths = []
        self.external_file_filter_mode = 0


def read_config_file(path='./config.ini'):
    # set file names etc +++++++++++++++++++++++++++++++
    my_config = SpecNetGenConfig()

    inifile = configparser.SafeConfigParser()
    inifile.read(path)

    # ------------------------------
    # [output] section
    my_config.output_folder_path = inifile.get('output', 'output_folder_path')
    if not os.path.isdir(my_config.output_folder_path):
        os.makedirs(my_config.output_folder_path)

    my_config.output_filename = inifile.get('output', 'output_filename')

    # ------------------------------
    # [input] section
    my_config.input_sample_folder_path = inifile.get('input', 'input_sample_folder_path')
    if not os.path.isdir(my_config.input_sample_folder_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), my_config.input_sample_folder_path)

    _list_input_sample_extension = inifile.get('input', 'input_sample_extension').split(',')
    _list_input_sample_extension = [_e for _e in _list_input_sample_extension if _e]
    if not _list_input_sample_extension:
        _list_input_sample_extension = ['msp', 'mgf', 'json']
    for _ext in _list_input_sample_extension:
        _ext = _ext.strip()
        _ext = _ext.lstrip('.')
        if _ext:
            for _path in glob(f'{my_config.input_sample_folder_path}/*.{_ext}'):
                my_config.list_sample_file_path.append(_path)
                my_config.list_sample_filename.append(os.path.basename(_path))

    my_config.input_ref_folder_path = inifile.get('input', 'input_ref_folder_path')
    if not os.path.isdir(my_config.input_ref_folder_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), my_config.input_ref_folder_path)

    _list_input_ref_extension = inifile.get('input', 'input_ref_extension').split(',')
    _list_input_ref_extension = [_e for _e in _list_input_ref_extension if _e]
    if not _list_input_ref_extension:
        _list_input_ref_extension = ['msp', 'mgf', 'json']
    for _ext in _list_input_ref_extension:
        _ext = _ext.strip()
        _ext = _ext.lstrip('.')
        if _ext:
            for _path in glob(f'{my_config.input_ref_folder_path}/*.{_ext}'):
                my_config.list_ref_file_path.append(_path)
                my_config.list_ref_filename.append(os.path.basename(_path))

    my_config.input_blank_folder_path = inifile.get('input', 'input_blank_folder_path')
    if not os.path.isdir(my_config.input_blank_folder_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), my_config.input_blank_folder_path)

    _list_input_blank_extension = inifile.get('input', 'input_ref_extension').split(',')
    _list_input_blank_extension = [_e for _e in _list_input_blank_extension if _e]
    if not _list_input_blank_extension:
        _list_input_blank_extension = ['msp', 'mgf', 'json']
    for _ext in _list_input_blank_extension:
        _ext = _ext.strip()
        _ext = _ext.lstrip('.')
        if _ext:
            for _path in glob(f'{my_config.input_blank_folder_path}/*.{_ext}'):
                my_config.list_blank_file_path.append(_path)
                my_config.list_blank_filename.append(os.path.basename(_path))

    _list_ref_spec_filename = inifile.get('input', 'ref_spec_filename').split(',')
    for _filename in _list_ref_spec_filename:
        _filename = _filename.strip()
        if not _filename:
            continue
        _path = os.path.join(my_config.input_ref_folder_path, _filename)
        if not os.path.isfile(_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _path)
        my_config.list_ref_spec_filename.append(_filename)
        my_config.list_ref_spec_path.append(_path)

    my_config.create_edge_within_layer_ref = int(inifile.get('input', 'create_edge_within_layer_ref'))

    _ref_split_category = inifile.get('input', 'ref_split_category')
    if _ref_split_category not in ('', 'cmpd_classification_superclass', 'cmpd_classification_class', 'cmpd_pathway'):
        raise ValueError('"ref_split_category" must be "cmpd_classification_superclass", "cmpd_classification_class" '
                         'or "cmpd_pathway".')
    my_config.ref_split_category = _ref_split_category

    _list_ref_select_keyword = inifile.get('input', 'ref_select_keywords').split(',')
    my_config.list_ref_select_keyword = [_key.strip() for _key in _list_ref_select_keyword if _key.strip()]

    _list_ref_exclude_keyword = inifile.get('input', 'ref_exclude_keywords').split(',')
    my_config.list_ref_exclude_keyword = [_key.strip() for _key in _list_ref_exclude_keyword if _key.strip()]

    _list_decoy = inifile.get('input', 'decoy').split(',')
    my_config.list_decoy = [_decoy.strip() for _decoy in _list_decoy if _decoy.strip()]

    # ------------------------------
    # [filter] section
    _list_author = inifile.get('filter', 'authors').split(';')
    my_config.list_author = [_author.strip() for _author in _list_author if _author.strip()]

    _list_characters = inifile.get('filter', 'name_key_characters_to_remove').split(',')
    my_config.list_name_key_characters_to_remove = [_character.strip() for _character in _list_characters
                                                    if _character.strip()]

    my_config.instrument_type = inifile.get('filter', 'instrument_type')
    my_config.ionization = inifile.get('filter', 'ionization')

    _list_precursor_type = inifile.get('filter', 'precursor_type').split(',')
    my_config.list_precursor_type = [_precursor_type.strip() for _precursor_type in _list_precursor_type
                                     if _precursor_type.strip()]

    my_config.fragmentation_type = inifile.get('filter', 'fragmentation_type')

    my_config.min_number_of_peaks = int(inifile.get('filter', 'min_number_of_peaks'))

    _list_compound_dat = inifile.get('filter', 'filename_compound_dat_for_filter').split(',')
    my_config.list_filename_compound_dat_for_filter = [_dat.strip() for _dat in _list_compound_dat if _dat.strip()]

    my_config.remove_spec_wo_prec_mz = int(inifile.get('filter', 'remove_spec_wo_prec_mz'))

    _list_filename_avoid_filter = inifile.get('filter', 'filename_avoid_filter').split(',')
    my_config.list_filename_avoid_filter = [_filename.strip() for _filename in _list_filename_avoid_filter
                                            if _filename.strip()]

    _list_filename_top_X_peak_rich = inifile.get('filter', 'filename_top_X_peak_rich').split(',')
    my_config.list_filename_top_X_peak_rich = [_filename.strip() for _filename in _list_filename_top_X_peak_rich
                                               if _filename.strip()]

    my_config.num_top_X_peak_rich = int(inifile.get('filter', 'num_top_X_peak_rich'))

    # ------------------------------
    # [spectrum processing] section
    _list_remove_low_intensity_peaks = inifile.get('spectrum processing', 'remove_low_intensity_peaks').split(',')
    list_remove_low_intensity_peaks = [float(_value.strip()) for _value in _list_remove_low_intensity_peaks
                                       if _value.strip()]

    _list_deisotope_width = inifile.get('spectrum processing', 'deisotope_width').split(',')
    list_deisotope_width = [float(_value.strip()) for _value in _list_deisotope_width if _value.strip()]

    _list_deisotope_int_ratio = inifile.get('spectrum processing', 'deisotope_int_ratio').split(',')
    list_deisotope_int_ratio = [float(_value.strip()) for _value in _list_deisotope_int_ratio if _value.strip()]

    _list_topN_binned_ranges_topN_number = inifile.get('spectrum processing',
                                                       'topN_binned_ranges_topN_number').split(',')
    list_topN_binned_ranges_topN_number = [int(_value.strip()) for _value in _list_topN_binned_ranges_topN_number
                                           if _value.strip()]

    _list_topN_binned_ranges_bin_size = inifile.get('spectrum processing', 'topN_binned_ranges_bin_size').split(',')
    list_topN_binned_ranges_bin_size = [int(_value.strip()) for _value in _list_topN_binned_ranges_bin_size
                                        if _value.strip()]
    _list_intensity_convert_mode = inifile.get('spectrum processing', 'intensity_convert_mode').split(',')
    list_intensity_convert_mode = [int(_value.strip()) for _value in _list_intensity_convert_mode if _value.strip()]

    my_config.make_spec_nr_in_ds = int(inifile.get('spectrum processing', 'make_spec_nr_in_ds'))

    # ------------------------------
    # [peak matching related] section
    _list_spec_matching_mode = inifile.get('peak matching related', 'spec_matching_mode').split(',')
    list_spec_matching_mode = [int(_value.strip()) for _value in _list_spec_matching_mode if _value.strip()]

    _list_mz_tol = inifile.get('peak matching related', 'mz_tol').split(',')
    list_mz_tol = [float(_value.strip()) for _value in _list_mz_tol if _value.strip()]

    _list_matching_top_N_input = inifile.get('peak matching related', 'matching_top_N_input').split(',')
    list_matching_top_N_input = [int(_value.strip()) for _value in _list_matching_top_N_input if _value.strip()]

    _list_match_peaklist_length = inifile.get('peak matching related', 'match_peaklist_length').split(',')
    list_match_peaklist_length = [int(_value.strip()) for _value in _list_match_peaklist_length if _value.strip()]

    # ------------------------------
    # [threshold] section
    _list_score_threshold_to_output = inifile.get('threshold', 'score_threshold').split(',')
    list_score_threshold_to_output = [float(_value.strip()) for _value in _list_score_threshold_to_output
                                      if _value.strip()]
    _list_minimum_peak_match_to_output = inifile.get('threshold', 'minimum_peak_match_to_output').split(',')
    list_minimum_peak_match_to_output = [int(_value.strip()) for _value in _list_minimum_peak_match_to_output
                                         if _value.strip()]

    _list_min_num_input_peaks_for_valid_bonanza = inifile.get('threshold',
                                                              'minimum_num_input_peak_for_bonanza').split(',')
    list_min_num_input_peaks_for_valid_bonanza = [
        int(_value.strip()) for _value in _list_min_num_input_peaks_for_valid_bonanza if _value.strip()]

    # ------------------------------
    # [external info files] section
    my_config.compound_table_folder_path = inifile.get('external info files', 'compound_table_folder_path')
    for _f in os.listdir(my_config.compound_table_folder_path):
        if (os.path.isfile(os.path.join(my_config.compound_table_folder_path, _f))
                and os.path.splitext(_f)[1] in ('.tsv', '.csv')):
            my_config.compound_table_paths.append(os.path.join(my_config.compound_table_folder_path, _f))

    my_config.filename_metacyc_cmpd_dat = inifile.get('external info files', 'metacyc_compound_dat')
    my_config.metacyc_cmpd_dat_path = os.path.join(my_config.compound_table_folder_path,
                                                   my_config.filename_metacyc_cmpd_dat)
    if not os.path.isfile(my_config.metacyc_cmpd_dat_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), my_config.metacyc_cmpd_dat_path)

    my_config.filename_metacyc_pathway_dat = inifile.get('external info files', 'metacyc_pathway_dat')
    my_config.metacyc_pathway_dat_path = os.path.join(my_config.compound_table_folder_path,
                                                      my_config.filename_metacyc_pathway_dat)
    if not os.path.isfile(my_config.metacyc_pathway_dat_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), my_config.metacyc_pathway_dat_path)

    my_config.external_file_filter_mode = int(inifile.get('external info files', 'mode'))

    # ------------------------------
    # [assessment] section
    my_config.class_matching_correlation = int(inifile.get('assessment', 'class_matching_correlation'))

    # ------------------------------
    # [other] section
    my_config.flag_write_log = int(inifile.get('other', 'write log'))
    my_config.flag_display_process = int(inifile.get('other', 'display process'))

    config_filename = os.path.basename(path)
    shutil.copyfile(path, os.path.join(my_config.output_folder_path, config_filename))

    # ----------------------------------------
    # Create combination of config settings
    list_header_df = ['idx_conf', 'remove_low_intensity_peaks', 'deisotope_width', 'deisotope_int_ratio',
                      'topN_binned_ranges_topN_number', 'topN_binned_ranges_bin_size', 'intensity_convert_mode',
                      'spec_matching_mode', 'mz_tol', 'matching_top_N_input', 'match_peaklist_length',
                      'score_threshold_to_output', 'minimum_peak_match_to_output',
                      'min_num_input_peaks_for_valid_bonanza']

    df_conf = pd.DataFrame(columns=list_header_df)

    list_config = []
    idx_conf = -1

    for (remove_low_intensity_peaks, deisotope_width, deisotope_int_ratio, topN_binned_ranges_topN_number,
         topN_binned_ranges_bin_size, intensity_convert_mode, spec_matching_mode, mz_tol, matching_top_N_input,
         match_peaklist_length, score_threshold_to_output, minimum_peak_match_to_output,
         min_num_input_peaks_for_valid_bonanza)\
        in itertools.product(list_remove_low_intensity_peaks, list_deisotope_width, list_deisotope_int_ratio,
                             list_topN_binned_ranges_topN_number, list_topN_binned_ranges_bin_size,
                             list_intensity_convert_mode, list_spec_matching_mode, list_mz_tol,
                             list_matching_top_N_input, list_match_peaklist_length, list_score_threshold_to_output,
                             list_minimum_peak_match_to_output, list_min_num_input_peaks_for_valid_bonanza):
        idx_conf += 1

        df_conf.loc[idx_conf, :] = [
            idx_conf, remove_low_intensity_peaks, deisotope_width, deisotope_int_ratio, topN_binned_ranges_topN_number,
            topN_binned_ranges_bin_size, intensity_convert_mode, spec_matching_mode, mz_tol, matching_top_N_input,
            match_peaklist_length, score_threshold_to_output, minimum_peak_match_to_output,
            min_num_input_peaks_for_valid_bonanza
        ]

        obj_config = deepcopy(my_config)

        # id conf
        obj_config.id = idx_conf
        obj_config.remove_low_intensity_peaks = remove_low_intensity_peaks
        obj_config.deisotope_width = deisotope_width
        obj_config.deisotope_int_ratio = deisotope_int_ratio
        obj_config.topN_binned_ranges_topN_number = topN_binned_ranges_topN_number
        obj_config.topN_binned_ranges_bin_size = topN_binned_ranges_bin_size
        obj_config.intensity_convert_mode = intensity_convert_mode
        obj_config.spec_matching_mode = spec_matching_mode
        obj_config.mz_tol = float(mz_tol)
        obj_config.matching_top_N_input = matching_top_N_input
        obj_config.match_peaklist_length = match_peaklist_length
        obj_config.score_threshold_to_output = score_threshold_to_output
        obj_config.minimum_peak_match_to_output = minimum_peak_match_to_output
        obj_config.min_num_input_peaks_for_valid_bonanza = min_num_input_peaks_for_valid_bonanza
        list_config.append(obj_config)

    df_conf.to_csv(os.path.join(my_config.output_folder_path, 'config_combination.csv'))

    return list_config
