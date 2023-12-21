__version__ = 'a6'
import logging
import pickle
from logging import DEBUG, FileHandler, Formatter, getLogger, INFO, StreamHandler
import h5py
import os
from my_filter import extract_top_x_peak_rich, filter_reference_spectra, remove_blank_spectra_from_sample_spectra
from my_parser import metacyc_parser as read_meta
from my_parser.cluster_attribute_parser import write_cluster_attribute
from my_parser.edge_info_parser import write_edge_info
from my_parser.matchms_spectrum_parser import (get_serialized_spectra_paths,
                                               initialize_serialize_spectra_file,
                                               load_and_serialize_spectra,
                                               serialize_filtered_spectra,
                                               serialize_grouped_spectra)
from my_parser.score_parser import initialize_score_files, initialize_score_hdf5
from my_parser.spectrum_metadata_parser import (group_by_dataset,
                                                initialize_spectrum_metadata_file,
                                                initialize_spectrum_metadata_hdf5,
                                                write_metadata)
from score.score import calculate_similarity_score, clustering_based_on_inchikey
from utils import add_compound_info, add_dataset_keyword, add_metacyc_compound_info, check_filtered_metadata
from utils.clustering import create_cluster_frame


LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


def generate_spectral_network(config_obj, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER
    
    logger.info('started')

    initialize_serialize_spectra_file()
    initialize_spectrum_metadata_file()

    # Remove metacyc files -----------------------------------------
    metacyc_files = {'compound': './metacyc_compound.npy',
                     'pathway': './metacyc_pathway.npy',
                     'filter': './metacyc_compound_for_filter.npy'}
    for _p in metacyc_files.values():
        if os.path.isfile(_p):
            os.remove(_p)
    # --------------------------------------------------------------

    initialize_score_hdf5()
    initialize_score_files()

    # make sure which type/version of spec obj you are using
    # input variable config_obj.id is used when multiple versions of config object is used for mainly cross validation
    if config_obj.id > -1:
        output_folder_name = os.path.basename(os.path.dirname(config_obj.output_folder_path))
        config_obj.output_folder_path = os.path.join(config_obj.output_folder_path,
                                                     f'config_id_{config_obj.id}')

        if not os.path.isdir(config_obj.output_folder_path):
            os.makedirs(config_obj.output_folder_path)
    else:
        raise ValueError(f'config_obj.id must be >= 0: {config_obj.id}')

    # create a logger to export a report.
    report = getLogger('report')
    report.setLevel(INFO)
    report_handler = FileHandler(os.path.join(config_obj.output_folder_path, f'log_{config_obj.id}.txt'))
    report_handler.setLevel(INFO)
    report_formatter = Formatter('%(message)s')
    report_handler.setFormatter(report_formatter)
    report.addHandler(report_handler)

    report.info(f'config_obj.id is {config_obj.id}')

    # ------------------
    # output file names
    # ------------------
    cluster_attribute_path = os.path.join(config_obj.output_folder_path,
                                          f'{config_obj.output_filename}.cluster_attribute.tsv')
    edge_info_path = os.path.join(config_obj.output_folder_path, f'{config_obj.output_filename}.edgeinfo.tsv')

    # -----------------------------
    # Read spectral data files
    # -----------------------------
    logger.debug("Starting [C] reading spectral data files")

    if config_obj.flag_write_log:
        report.info('start reading data files ...\n')

    source_filename_list = []
    source_filename_vs_tag_id = {}
    source_filename_vs_serialized_spectra_dir = {}

    spectrum_index = 0

    # Read and serialize sample spectra. --------------------------------------------------------------
    for _idx, _path in enumerate(config_obj.list_sample_file_path):
        _filename = os.path.basename(_path)
        source_filename_list.append(_filename)
        source_filename_vs_tag_id[_filename] = f'sample_{_idx}'

        # Directory to export serialized spectra
        _serialized_spectra_dir = f'./serialized_spectra/raw/{source_filename_vs_tag_id[_filename]}'
        source_filename_vs_serialized_spectra_dir[_filename] = _serialized_spectra_dir

        # Whether to introduce mass shift
        _is_random_mass_shift_enabled = False
        if _filename in config_obj.list_decoy or 'all' in config_obj.list_decoy:
            _is_random_mass_shift_enabled = True

        spectrum_index = load_and_serialize_spectra(
            _path, 'sample', _serialized_spectra_dir, spectrum_index,
            intensity_threshold=config_obj.remove_low_intensity_peaks,
            is_introduce_random_mass_shift=_is_random_mass_shift_enabled,
            deisotope_int_ratio=config_obj.deisotope_int_ratio,
            deisotope_mz_tol=config_obj.deisotope_mz_tol,
            binning_top_n=config_obj.top_n_binned_ranges_top_n_number,
            binning_range=config_obj.top_n_binned_ranges_bin_size,
            matching_top_n_input=config_obj.matching_top_n_input)

        spectrum_index += 1

    # Read and serialize reference spectra. --------------------------------------------------------------
    for _idx, _path in enumerate(config_obj.list_ref_file_path):
        _filename = os.path.basename(_path)
        source_filename_list.append(_filename)
        source_filename_vs_tag_id[_filename] = f'ref_{_idx}'

        # Directory to export serialized spectra
        _serialized_spectra_dir = f'./serialized_spectra/raw/{source_filename_vs_tag_id[_filename]}'
        source_filename_vs_serialized_spectra_dir[_filename] = _serialized_spectra_dir

        # Whether to introduce mass shift
        _is_random_mass_shift_enabled = False
        if _filename in config_obj.list_decoy or 'all' in config_obj.list_decoy:
            _is_random_mass_shift_enabled = True

        spectrum_index = load_and_serialize_spectra(
            _path, 'ref', _serialized_spectra_dir, spectrum_index,
            intensity_threshold=config_obj.remove_low_intensity_peaks,
            is_introduce_random_mass_shift=_is_random_mass_shift_enabled,
            deisotope_int_ratio=config_obj.deisotope_int_ratio,
            deisotope_mz_tol=config_obj.deisotope_mz_tol,
            binning_top_n=config_obj.top_n_binned_ranges_top_n_number,
            binning_range=config_obj.top_n_binned_ranges_bin_size,
            matching_top_n_input=config_obj.matching_top_n_input)

        spectrum_index += 1

    # Read and serialize blank spectra. --------------------------------------------------------------
    for _idx, _path in enumerate(config_obj.list_blank_file_path):
        _filename = os.path.basename(_path)
        source_filename_list.append(_filename)
        source_filename_vs_tag_id[_filename] = f'blank_{_idx}'

        # Directory to export serialized spectra
        _serialized_spectra_dir = f'./serialized_spectra/raw/{source_filename_vs_tag_id[_filename]}'
        source_filename_vs_serialized_spectra_dir[_filename] = _serialized_spectra_dir

        # Whether to introduce mass shift
        _is_random_mass_shift_enabled = False
        if _filename in config_obj.list_decoy or 'all' in config_obj.list_decoy:
            _is_random_mass_shift_enabled = True

        spectrum_index = load_and_serialize_spectra(
            _path, 'blank', _serialized_spectra_dir, spectrum_index,
            intensity_threshold=config_obj.remove_low_intensity_peaks,
            is_introduce_random_mass_shift=_is_random_mass_shift_enabled,
            deisotope_int_ratio=config_obj.deisotope_int_ratio,
            deisotope_mz_tol=config_obj.deisotope_mz_tol,
            binning_top_n=config_obj.top_n_binned_ranges_top_n_number,
            binning_range=config_obj.top_n_binned_ranges_bin_size,
            matching_top_n_input=config_obj.matching_top_n_input)

        spectrum_index += 1

    logger.info(f'Number of all spectra: {spectrum_index}')

    # Write metadata of sample spectra --------------------------------------------------------------
    for _filename in config_obj.list_sample_filename:
        _serialized_spectra_dir = source_filename_vs_serialized_spectra_dir[_filename]
        _serialized_spectra_paths = get_serialized_spectra_paths(_serialized_spectra_dir)

        for _serialized_spectra_path, _, _ in _serialized_spectra_paths:
            logger.info(f'Write metadata of spectra in {os.path.basename(_serialized_spectra_path)}')
            with open(_serialized_spectra_path, 'rb') as f:
                _spectra = pickle.load(f)

            write_metadata('./spectrum_metadata/raw/sample_metadata.npy', _spectra, export_tsv=True)

    # Write metadata of reference spectra --------------------------------------------------------------
    for _filename in config_obj.list_ref_filename:
        _serialized_spectra_dir = source_filename_vs_serialized_spectra_dir[_filename]
        _serialized_spectra_paths = get_serialized_spectra_paths(_serialized_spectra_dir)

        for _serialized_spectra_path, _, _ in _serialized_spectra_paths:
            logger.info(f'Write metadata of spectra in {os.path.basename(_serialized_spectra_path)}')
            with open(_serialized_spectra_path, 'rb') as f:
                _spectra = pickle.load(f)

            write_metadata('./spectrum_metadata/raw/ref_metadata.npy', _spectra, export_tsv=True)

    # Write metadata of blank spectra --------------------------------------------------------------
    for _filename in config_obj.list_blank_filename:
        _serialized_spectra_dir = source_filename_vs_serialized_spectra_dir[_filename]
        _serialized_spectra_paths = get_serialized_spectra_paths(_serialized_spectra_dir)

        for _serialized_spectra_path, _, _ in _serialized_spectra_paths:
            logger.info(f'Write metadata of spectra in {os.path.basename(_serialized_spectra_path)}')
            with open(_serialized_spectra_path, 'rb') as f:
                _spectra = pickle.load(f)

            write_metadata('./spectrum_metadata/raw/blank_metadata.npy', _spectra, export_tsv=True)

    # Remove common contaminants in sample data by subtracting blank elements ---------------
    remove_blank_spectra_from_sample_spectra(
        './spectrum_metadata/raw/blank_metadata.npy', './spectrum_metadata/raw/sample_metadata.npy',
        './spectrum_metadata/filtered/sample_metadata.npy',
        mz_tolerance=config_obj.mz_tol_to_remove_blank, rt_tolerance=config_obj.rt_tol_to_remove_blank,
        export_tsv=True)

    # Check whether there are remaining spectra after filtering.
    if not check_filtered_metadata('./spectrum_metadata/filtered/sample_metadata.npy'):
        raise ValueError('There is no spectrum after filtering.')

    # ------------------------------------------------
    # Add external compound data to reference spectra
    # ------------------------------------------------
    if config_obj.compound_table_paths:
        add_compound_info(config_obj.compound_table_paths,
                          ['./spectrum_metadata/filtered/sample_metadata.npy',
                           './spectrum_metadata/raw/ref_metadata.npy'])

    if config_obj.metacyc_cmpd_dat_path:
        read_meta.convert_metacyc_compounds_dat_to_npy(config_obj.metacyc_cmpd_dat_path,
                                                       output_path=metacyc_files['compound'],
                                                       parameters_to_open_file=dict(encoding='utf8', errors='replace'))
    if config_obj.metacyc_pathway_dat_path:
        read_meta.convert_metacyc_pathways_dat_to_npy(config_obj.metacyc_pathway_dat_path,
                                                      output_path=metacyc_files['pathway'],
                                                      parameters_to_open_file=dict(encoding='utf8', errors='replace'))

    if config_obj.metacyc_cmpd_dat_path and config_obj.metacyc_pathway_dat_path:
        read_meta.assign_pathway_id_to_compound_in_npy(metacyc_files['compound'], metacyc_files['pathway'])
    
    add_metacyc_compound_info(metacyc_files['compound'], './spectrum_metadata/raw/ref_metadata.npy', export_tsv=True)
    
    # ------------------------------------
    # External compound file for filtering
    # ------------------------------------
    logger.debug('Read external compound file for filtering')
    for _path in config_obj.list_path_compound_dat_for_filter:
        logger.debug(f"filename: {os.path.basename(_path)}")
        read_meta.convert_metacyc_compounds_dat_to_npy(_path,
                                                       output_path=metacyc_files['filter'],
                                                       parameters_to_open_file=dict(encoding='utf8', errors='replace'))

    report.info(f"\nfiles NOT to be filtered {str(config_obj.list_filename_avoid_filter)}\n")
    filter_reference_spectra(config_obj,
                             ref_metadata_path='./spectrum_metadata/raw/ref_metadata.npy',
                             output_path='./spectrum_metadata/filtered/ref_metadata.npy',
                             metacyc_compound_path=metacyc_files['filter'],
                             export_tsv=True,
                             _logger=report)

    # ------------------------------------
    # Extract top X peak rich spectra
    # ------------------------------------
    if config_obj.num_top_x_peak_rich:
        for _filename in config_obj.list_sample_filename:
            extract_top_x_peak_rich(_filename, config_obj.num_top_x_peak_rich)
    
    serialize_filtered_spectra()

    # --------------------------------------
    # Group metadata and spectra by dataset
    # --------------------------------------
    add_dataset_keyword(config_obj.ref_split_category)
    group_by_dataset()
    serialize_grouped_spectra()

    # -------------------------------
    # Calculate spectral similarity
    # -------------------------------
    config_obj.is_clustering_required = create_cluster_frame()
    calculate_similarity_score(config_obj.spec_matching_mode, config_obj.mz_tol, config_obj.intensity_convert_mode)
    clustering_based_on_inchikey()

    # -------
    # Output
    # -------
    write_edge_info(edge_info_path, config_obj.score_threshold_to_output, config_obj.minimum_peak_match_to_output,
                    config_obj.mz_tol, config_obj.create_edge_within_layer_ref)
    write_cluster_attribute(cluster_attribute_path, config_obj.ref_split_category)
