import configparser
from copy import deepcopy
from datetime import datetime
import errno
from glob import glob
import itertools
import logging
from logging import DEBUG, Formatter, getLogger, StreamHandler
import os
import pandas as pd
import shutil


LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


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
        self.input_calculated_ref_score_dir = ''
        self.reuse_serialized_reference_spectra = False

        self.list_ref_spec_filename = []
        self.list_ref_spec_path = []
        self.sample_split_category = ''
        self.ref_split_category = ''
        self.list_ref_select_keyword = []
        self.list_ref_exclude_keyword = []

        self.calculate_inner_sample = True
        self.calculate_inter_sample = True
        self.calculate_inter_sample_and_reference = True
        self.calculate_inner_reference = True
        self.calculate_inter_reference = True
        self.list_decoy = []

        # filtering
        self.list_author = ''
        self.list_name_key_characters_to_remove = []
        self.instrument_type = ''
        self.ion_mode = ''
        self.list_precursor_type = []
        self.fragmentation_type = ''
        self.ionization = ''
        self.min_number_of_peaks = 1
        self.list_path_compound_dat_for_filter = []
        self.remove_spec_wo_prec_mz = 0
        self.list_filename_avoid_filter = []
        self.num_top_x_peak_rich = 0

        # spectral processing
        self.mz_tol_to_remove_blank = 0
        self.rt_tol_to_remove_blank = 0
        self.remove_low_intensity_peaks = 0.0005
        self.deisotope_int_ratio = 3
        self.deisotope_mz_tol = 0
        self.top_n_binned_ranges_top_n_number = 10
        self.top_n_binned_ranges_bin_size = 10
        self.intensity_convert_mode = 0

        # matching related
        self.spec_matching_mode = 1
        self.mz_tol = 0
        self.matching_top_n_input = -1

        self.score_threshold_to_output = 0
        self.minimum_peak_match_to_output = 0

        self.output_filename = ''
        self.output_folder_path = ''
        self.fo_score_vs_chemsim = ''

        self.export_reference_score = False
        self.output_ref_score_dir_path = ''
        self.export_serialized_reference_spectra = False

        self.class_matching_correlation = 0

        self.flag_write_log = 0
        self.flag_display_process = 0

        self.filename_metacyc_cmpd_dat = ''
        self.filename_metacyc_pathway_dat = ''
        self.compound_table_folder_path = ''
        self.compound_table_paths = []
        self.external_file_filter_mode = 0
        self.metacyc_cmpd_dat_path = ''
        self.metacyc_pathway_dat_path = ''

        # Parameters not defined by a user
        self.is_clustering_required = False
        self.max_length_of_cluster_id = 100


def read_config_file(path='./config.ini', _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER
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

    export_reference_score = inifile.get('output', 'export_reference_score')
    if export_reference_score not in ('0', '1'):
        raise ValueError(f'"export_reference_score" must be 0 or 1: {export_reference_score}')
    if export_reference_score == '0':
        my_config.export_reference_score = False
    else:
        my_config.export_reference_score = True

    export_serialized_reference_spectra = inifile.get('output', 'export_serialized_reference_spectra')
    if export_serialized_reference_spectra not in ('0', '1'):
        raise ValueError(f'"export_serialized_reference_spectra" must be 0 or 1: {export_serialized_reference_spectra}')
    if export_serialized_reference_spectra == '0':
        my_config.export_serialized_reference_spectra = False
    else:
        my_config.export_serialized_reference_spectra = True
        my_config.export_reference_score = True

    # Make a folder to export reference score files.
    if my_config.export_reference_score:
        now = datetime.now()
        my_config.output_ref_score_dir_path = f"./ref_score_{now.strftime('%Y%m%d%H%M%S')}"
        if not os.path.isdir(my_config.output_ref_score_dir_path):
            os.makedirs(my_config.output_ref_score_dir_path)

    # ------------------------------
    # [input] section

    # sample files ------------------------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------------------------------------

    # blank files -------------------------------------------------------------------------------------------
    my_config.input_blank_folder_path = inifile.get('input', 'input_blank_folder_path')
    if my_config.input_blank_folder_path and not os.path.isdir(my_config.input_blank_folder_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), my_config.input_blank_folder_path)

    _list_input_blank_extension = inifile.get('input', 'input_blank_extension').split(',')
    _list_input_blank_extension = [_e for _e in _list_input_blank_extension if _e]
    if not _list_input_blank_extension:
        _list_input_blank_extension = ['msp', 'mgf', 'json']

    if my_config.input_blank_folder_path:
        for _ext in _list_input_blank_extension:
            _ext = _ext.strip()
            _ext = _ext.lstrip('.')
            if _ext:
                for _path in glob(f'{my_config.input_blank_folder_path}/*.{_ext}'):
                    my_config.list_blank_file_path.append(_path)
                    my_config.list_blank_filename.append(os.path.basename(_path))
    # -------------------------------------------------------------------------------------------------------

    # reference files ---------------------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------------------------------------

    _list_decoy = inifile.get('input', 'decoy').split(',')
    my_config.list_decoy = [_decoy.strip() for _decoy in _list_decoy if _decoy.strip()]

    # reuse reference score ---------------------------------------------------------------------------------
    my_config.input_calculated_ref_score_dir = inifile.get('input', 'input_calculated_ref_score_dir')
    reuse_config_path = ''
    if my_config.input_calculated_ref_score_dir:
        # Check whether config.ini exists in input_calculated_ref_score_dir.
        reuse_config_path = os.path.join(my_config.input_calculated_ref_score_dir, 'config.ini')
        if not os.path.isfile(reuse_config_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), reuse_config_path)

    # reuse serialized reference spectra
    reuse_serialized_reference_spectra = inifile.get('input', 'reuse_serialized_reference_spectra')
    if reuse_serialized_reference_spectra == '0':
        my_config.reuse_serialized_reference_spectra = False
    else:
        my_config.reuse_serialized_reference_spectra = True

        if not my_config.input_calculated_ref_score_dir:
            raise ValueError('Set a path to "input_calculated_ref_score_dir"')

    if my_config.input_calculated_ref_score_dir:
        # !!! Use previous settings. !!!!!!!!!!!!
        inifile = configparser.SafeConfigParser()
        inifile.read(reuse_config_path)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # -------------------------------------------------------------------------------------------------------

    # ------------------------------
    # [dataset] section
    calculate_inner_sample = inifile.get('dataset', 'calculate_inner_sample')
    if calculate_inner_sample not in ('0', '1'):
        raise ValueError(f'"calculate_inner_sample" must be 0 or 1: {calculate_inner_sample}')
    if calculate_inner_sample == '0':
        my_config.calculate_inner_sample = False
    else:
        my_config.calculate_inner_sample = True

    calculate_inter_sample = inifile.get('dataset', 'calculate_inter_sample')
    if calculate_inter_sample not in ('0', '1'):
        raise ValueError(f'"calculate_inter_sample" must be 0 or 1: {calculate_inter_sample}')
    if calculate_inter_sample == '0':
        my_config.calculate_inter_sample = False
    else:
        my_config.calculate_inter_sample = True

    calculate_inter_sample_and_reference = inifile.get('dataset', 'calculate_inter_sample_and_reference')
    if calculate_inter_sample_and_reference not in ('0', '1'):
        raise ValueError(f'"calculate_inter_sample_and_reference" must be 0 or 1: {calculate_inter_sample_and_reference}')
    if calculate_inter_sample_and_reference == '0':
        my_config.calculate_inter_sample_and_reference = False
    else:
        my_config.calculate_inter_sample_and_reference = True

    calculate_inner_reference = inifile.get('dataset', 'calculate_inner_reference')
    if calculate_inner_reference not in ('0', '1'):
        raise ValueError(f'"calculate_inner_reference" must be 0 or 1: {calculate_inner_reference}')
    if calculate_inner_reference == '0':
        my_config.calculate_inner_reference = False
    else:
        my_config.calculate_inner_reference = True

    calculate_inter_reference = inifile.get('dataset', 'calculate_inter_reference')
    if calculate_inter_reference not in ('0', '1'):
        raise ValueError(f'"calculate_inter_reference" must be 0 or 1: {calculate_inter_reference}')
    if calculate_inter_reference == '0':
        my_config.calculate_inter_reference = False
    else:
        my_config.calculate_inter_reference = True

    if my_config.input_calculated_ref_score_dir:
        my_config.calculate_inner_reference = False
        my_config.calculate_inter_reference = False

    _ref_split_category = inifile.get('dataset', 'ref_split_category')
    if _ref_split_category not in ('', 'source_filename','cmpd_classification_superclass', 'cmpd_classification_class', 'cmpd_pathway'):
        raise ValueError('"ref_split_category" must be "cmpd_classification_superclass", "cmpd_classification_class", '
                         '"cmpd_pathway", "source_filename" or "".')
    my_config.ref_split_category = _ref_split_category

    _sample_split_category = inifile.get('dataset', 'sample_split_category')
    if _sample_split_category not in ('', 'source_filename'):
        raise ValueError('"sample_split_category" must be "" or "source_filename"')
    my_config.sample_split_category = _sample_split_category

    _list_ref_select_keyword = inifile.get('dataset', 'ref_select_keywords').split(';')
    my_config.list_ref_select_keyword = [_key.strip() for _key in _list_ref_select_keyword if _key.strip()]

    _list_ref_exclude_keyword = inifile.get('dataset', 'ref_exclude_keywords').split(';')
    my_config.list_ref_exclude_keyword = [_key.strip() for _key in _list_ref_exclude_keyword if _key.strip()]

    # ------------------------------
    # [filter] section
    _list_author = inifile.get('filter', 'authors').split(';')
    my_config.list_author = [_author.strip() for _author in _list_author if _author.strip()]

    _list_characters = inifile.get('filter', 'name_key_characters_to_remove').split(',')
    my_config.list_name_key_characters_to_remove = [_character.strip() for _character in _list_characters
                                                    if _character.strip()]

    my_config.instrument_type = inifile.get('filter', 'instrument_type')
    my_config.ion_mode = inifile.get('filter', 'ion_mode').lower()
    my_config.ionization = inifile.get('filter', 'ionization')

    _list_precursor_type = inifile.get('filter', 'precursor_type').split(',')
    my_config.list_precursor_type = [_precursor_type.strip() for _precursor_type in _list_precursor_type
                                     if _precursor_type.strip()]

    my_config.fragmentation_type = inifile.get('filter', 'fragmentation_type')
    
    _min_number_of_peaks = inifile.get('filter', 'min_number_of_peaks')
    if _min_number_of_peaks:
        my_config.min_number_of_peaks = int(_min_number_of_peaks)
        if my_config.min_number_of_peaks < 1:
            raise ValueError(f'min_number_of_peaks must be larger than or equal to 1: {my_config.min_number_of_peaks}')

    _list_compound_dat = inifile.get('filter', 'path_of_compound_dat_for_filter').split(',')
    my_config.list_path_compound_dat_for_filter = [_dat.strip() for _dat in _list_compound_dat if _dat.strip()]

    _remove_spec_wo_prec_mz = inifile.get('filter', 'remove_spec_wo_prec_mz')
    if _remove_spec_wo_prec_mz:
        my_config.remove_spec_wo_prec_mz = int(_remove_spec_wo_prec_mz)
        if my_config.remove_spec_wo_prec_mz not in (0, 1):
            raise ValueError(f'min_number_of_peaks must be 0 or 1: {my_config.remove_spec_wo_prec_mz}')

    _list_filename_avoid_filter = inifile.get('filter', 'filename_avoid_filter').split(',')
    my_config.list_filename_avoid_filter = [_filename.strip() for _filename in _list_filename_avoid_filter
                                            if _filename.strip()]
    _num_top_x_peak_rich = inifile.get('filter', 'num_top_X_peak_rich')
    if _num_top_x_peak_rich:
        my_config.num_top_x_peak_rich= int(_num_top_x_peak_rich)
        if my_config.num_top_x_peak_rich < 0:
            raise ValueError(f'num_top_X_peak_rich must be larger than or equal to 0: {my_config.num_top_x_peak_rich}')
    else:
        my_config.num_top_x_peak_rich = 0

    # ------------------------------
    # [spectrum processing] section
    _list_mz_tol_to_remove_blank = inifile.get('spectrum processing', 'mz_tol_to_remove_blank').split(',')
    list_mz_tol_to_remove_blank = [float(_value.strip()) for _value in _list_mz_tol_to_remove_blank if _value.strip()]
    for _n in list_mz_tol_to_remove_blank:
        if _n < 0:
            raise ValueError(f'mz_tol_to_remove_blank must be larger than or equal to 0: {_n}')

    _list_rt_tol_to_remove_blank = inifile.get('spectrum processing', 'rt_tol_to_remove_blank').split(',')
    list_rt_tol_to_remove_blank = [float(_value.strip()) for _value in _list_rt_tol_to_remove_blank if _value.strip()]
    for _n in list_rt_tol_to_remove_blank:
        if _n < 0:
            raise ValueError(f'rt_tol_to_remove_blank must be larger than  or equal to 0: {_n}')

    _list_remove_low_intensity_peaks = inifile.get('spectrum processing', 'remove_low_intensity_peaks').split(',')
    list_remove_low_intensity_peaks = [float(_value.strip()) for _value in _list_remove_low_intensity_peaks
                                       if _value.strip()]
    for _n in list_remove_low_intensity_peaks:
        if _n < 0 or _n > 1:
            raise ValueError(f'remove_low_intensity_peaks must be in 0 to 1: {_n}')

    _list_deisotope_int_ratio = inifile.get('spectrum processing', 'deisotope_int_ratio').split(',')
    list_deisotope_int_ratio = [float(_value.strip()) for _value in _list_deisotope_int_ratio if _value.strip()]
    for _n in list_deisotope_int_ratio:
        if _n <= 0 and _n != -1:
            raise ValueError(f'deisotope_int_ratio must be larger than 0 or equal to -1: {_n}')    

    _list_deisotope_mz_tol = inifile.get('spectrum processing', 'deisotope_mz_tol').split(',')
    list_deisotope_mz_tol = [float(_value.strip()) for _value in _list_deisotope_mz_tol if _value.strip()]
    for _n in list_deisotope_mz_tol:
        if _n < 0:
            raise ValueError(f'deisotope_mz_tol must be larger than or equal to 0: {_n}')

    _list_top_n_binned_ranges_top_n_number = inifile.get('spectrum processing',
                                                         'topN_binned_ranges_topN_number').split(',')
    list_top_n_binned_ranges_top_n_number = [int(_value.strip()) for _value in _list_top_n_binned_ranges_top_n_number
                                             if _value.strip()]
    for _n in list_top_n_binned_ranges_top_n_number:
        if _n <= 0:
            raise ValueError(f'topN_binned_ranges_topN_number must be larger than 0: {_n}')

    _list_top_n_binned_ranges_bin_size = inifile.get('spectrum processing', 'topN_binned_ranges_bin_size').split(',')
    list_top_n_binned_ranges_bin_size = [float(_value.strip()) for _value in _list_top_n_binned_ranges_bin_size
                                         if _value.strip()]
    for _n in list_top_n_binned_ranges_bin_size:
        if _n <= 0 and _n != -1:
            raise ValueError(f'topN_binned_ranges_bin_size must be larger than 0 or equal to -1: {_n}')
    
    _list_intensity_convert_mode = inifile.get('spectrum processing', 'intensity_convert_mode').split(',')
    list_intensity_convert_mode = [int(_value.strip()) for _value in _list_intensity_convert_mode if _value.strip()]

    my_config.make_spec_nr_in_ds = int(inifile.get('spectrum processing', 'make_spec_nr_in_ds'))

    # ------------------------------
    # [peak matching related] section
    _list_spec_matching_mode = inifile.get('peak matching related', 'spec_matching_mode').split(',')
    list_spec_matching_mode = [int(_value.strip()) for _value in _list_spec_matching_mode if _value.strip()]
    for _n in list_spec_matching_mode:
        if _n not in (1, 2):
            raise ValueError(f'spec_matching_mode should be 1 or 2: {_n}')

    _list_mz_tol = inifile.get('peak matching related', 'mz_tol').split(',')
    list_mz_tol = [float(_value.strip()) for _value in _list_mz_tol if _value.strip()]

    _list_matching_top_n_input = inifile.get('peak matching related', 'matching_top_N_input').split(',')
    list_matching_top_n_input = [int(_value.strip()) for _value in _list_matching_top_n_input if _value.strip()]
    for _n in list_matching_top_n_input:
        if _n <= 0 and _n != -1:
            raise ValueError(f'matching_top_N_input must be larger than 0 or equal to -1: {_n}')

    # ------------------------------
    # [threshold] section
    _list_score_threshold_to_output = inifile.get('threshold', 'score_threshold').split(',')
    list_score_threshold_to_output = [float(_value.strip()) for _value in _list_score_threshold_to_output
                                      if _value.strip()]
    _list_minimum_peak_match_to_output = inifile.get('threshold', 'minimum_peak_match_to_output').split(',')
    list_minimum_peak_match_to_output = [int(_value.strip()) for _value in _list_minimum_peak_match_to_output
                                         if _value.strip()]

    # ------------------------------
    # [external info files] section
    my_config.compound_table_folder_path = inifile.get('external info files', 'compound_table_folder_path')
    for _f in os.listdir(my_config.compound_table_folder_path):
        if (os.path.isfile(os.path.join(my_config.compound_table_folder_path, _f))
                and os.path.splitext(_f)[1] in ('.tsv', '.csv')):
            my_config.compound_table_paths.append(os.path.join(my_config.compound_table_folder_path, _f))

    my_config.filename_metacyc_cmpd_dat = inifile.get('external info files', 'metacyc_compound_dat')
    if my_config.filename_metacyc_cmpd_dat:
        my_config.metacyc_cmpd_dat_path = os.path.join(my_config.compound_table_folder_path,
                                                       my_config.filename_metacyc_cmpd_dat)
        if not os.path.isfile(my_config.metacyc_cmpd_dat_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), my_config.metacyc_cmpd_dat_path)
    else:
        my_config.metacyc_cmpd_dat_path = ''

    my_config.filename_metacyc_pathway_dat = inifile.get('external info files', 'metacyc_pathway_dat')
    if my_config.filename_metacyc_pathway_dat:
        my_config.metacyc_pathway_dat_path = os.path.join(my_config.compound_table_folder_path,
                                                          my_config.filename_metacyc_pathway_dat)
        if not os.path.isfile(my_config.metacyc_pathway_dat_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), my_config.metacyc_pathway_dat_path)
    else:
        my_config.metacyc_pathway_dat_path = ''

    my_config.external_file_filter_mode = int(inifile.get('external info files', 'mode'))

    config_filename = os.path.basename(path)
    shutil.copyfile(path, os.path.join(my_config.output_folder_path, config_filename))

    if my_config.export_reference_score:
        shutil.copyfile(path, os.path.join(my_config.output_ref_score_dir_path, config_filename))

    # ----------------------------------------
    # Create combination of config settings
    list_header_df = ['idx_conf',
                      'mz_tol_to_remove_blank',
                      'rt_tol_to_remove_blank',
                      'remove_low_intensity_peaks',
                      'deisotope_int_ratio',
                      'deisotope_mz_tol',
                      'topN_binned_ranges_topN_number',
                      'topN_binned_ranges_bin_size',
                      'intensity_convert_mode',
                      'spec_matching_mode',
                      'mz_tol',
                      'matching_top_N_input',
                      'score_threshold_to_output',
                      'minimum_peak_match_to_output']

    df_conf = pd.DataFrame(columns=list_header_df)

    list_config = []
    idx_conf = -1

    for (mz_tol_to_remove_blank,
         rt_tol_to_remove_blank,
         remove_low_intensity_peaks,
         deisotope_int_ratio,
         deisotope_mz_tol,
         top_n_binned_ranges_top_n_number,
         top_n_binned_ranges_bin_size,
         intensity_convert_mode,
         spec_matching_mode,
         mz_tol,
         matching_top_n_input,
         score_threshold_to_output,
         minimum_peak_match_to_output)\
        in itertools.product(list_mz_tol_to_remove_blank,
                             list_rt_tol_to_remove_blank,
                             list_remove_low_intensity_peaks,
                             list_deisotope_int_ratio,
                             list_deisotope_mz_tol,
                             list_top_n_binned_ranges_top_n_number,
                             list_top_n_binned_ranges_bin_size,
                             list_intensity_convert_mode,
                             list_spec_matching_mode,
                             list_mz_tol,
                             list_matching_top_n_input,
                             list_score_threshold_to_output,
                             list_minimum_peak_match_to_output):
        idx_conf += 1

        df_conf.loc[idx_conf, :] = [
            idx_conf,
            mz_tol_to_remove_blank,
            rt_tol_to_remove_blank,
            remove_low_intensity_peaks,
            deisotope_int_ratio,
            deisotope_mz_tol,
            top_n_binned_ranges_top_n_number,
            top_n_binned_ranges_bin_size,
            intensity_convert_mode,
            spec_matching_mode,
            mz_tol,
            matching_top_n_input,
            score_threshold_to_output,
            minimum_peak_match_to_output
        ]

        obj_config = deepcopy(my_config)

        # id conf
        obj_config.id = idx_conf
        obj_config.mz_tol_to_remove_blank = mz_tol_to_remove_blank
        obj_config.rt_tol_to_remove_blank = rt_tol_to_remove_blank
        obj_config.remove_low_intensity_peaks = remove_low_intensity_peaks
        obj_config.deisotope_int_ratio = deisotope_int_ratio
        obj_config.deisotope_mz_tol = deisotope_mz_tol
        obj_config.top_n_binned_ranges_top_n_number = top_n_binned_ranges_top_n_number
        obj_config.top_n_binned_ranges_bin_size = top_n_binned_ranges_bin_size
        obj_config.intensity_convert_mode = intensity_convert_mode
        obj_config.spec_matching_mode = spec_matching_mode
        obj_config.mz_tol = mz_tol
        obj_config.matching_top_n_input = matching_top_n_input
        obj_config.score_threshold_to_output = score_threshold_to_output
        obj_config.minimum_peak_match_to_output = minimum_peak_match_to_output
        list_config.append(obj_config)

    df_conf.to_csv(os.path.join(my_config.output_folder_path, 'config_combination.csv'))

    return list_config
