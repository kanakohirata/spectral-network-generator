__version__ = 'a6'
import logging
from logging import DEBUG, FileHandler, Formatter, getLogger, INFO, StreamHandler
import h5py
import os
from filter.filter import extract_top_x_prak_rich, filter_reference_spectra, remove_blank_spectra_from_sample_spectra
from my_parser import metacyc_parser as read_meta
from my_parser.cluster_attribute_parser import write_cluster_attribute
from my_parser.edge_info_parser import write_edge_info
from my_parser.matchms_spectrum_parser import (delete_serialize_spectra_file, load_and_serialize_spectra,
                                               serialize_filtered_spectra)
from my_parser.score_parser import initialize_score_hdf5
from my_parser.spectrum_metadata_parser import initialize_spectrum_metadata_hdf5
from score.score import calculate_similarity_score, clustering_based_on_inchikey
from utils import add_classyfire_class, add_metacyc_compound_info
from utils.clustering import add_cluster_id


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

    if not os.path.isdir('./serialized_spectra'):
        os.makedirs('./serialized_spectra')
    if not os.path.isdir('./serialized_spectra/filtered'):
        os.makedirs('./serialized_spectra/filtered')

    delete_serialize_spectra_file()
    initialize_spectrum_metadata_hdf5()
    read_meta.initialize_metacyc_hdf5('./metacyc.h5')
    read_meta.initialize_metacyc_hdf5('./metacyc_for_filter.h5')
    initialize_score_hdf5()

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

    for _path in config_obj.list_sample_file_path:
        _filename = os.path.basename(_path)
        _is_introduce_random_mass_shift = False
        if _filename in config_obj.list_decoy or 'all' in config_obj.list_decoy:
            _is_introduce_random_mass_shift = True
        load_and_serialize_spectra(_path, 'sample', intensity_threshold=config_obj.remove_low_intensity_peaks,
                                   is_introduce_random_mass_shift=_is_introduce_random_mass_shift)
    for _path in config_obj.list_ref_file_path:
        _filename = os.path.basename(_path)
        _is_introduce_random_mass_shift = False
        if _filename in config_obj.list_decoy or 'all' in config_obj.list_decoy:
            _is_introduce_random_mass_shift = True
        load_and_serialize_spectra(_path, 'ref', intensity_threshold=config_obj.remove_low_intensity_peaks,
                                   is_introduce_random_mass_shift=_is_introduce_random_mass_shift)
    for _path in config_obj.list_blank_file_path:
        _filename = os.path.basename(_path)
        _is_introduce_random_mass_shift = False
        if _filename in config_obj.list_decoy or 'all' in config_obj.list_decoy:
            _is_introduce_random_mass_shift = True
        load_and_serialize_spectra(_path, 'blank', intensity_threshold=config_obj.remove_low_intensity_peaks,
                                   is_introduce_random_mass_shift=_is_introduce_random_mass_shift)

    # Remove common contaminants in sample data by subtracting blank elements ---------------
    remove_blank_spectra_from_sample_spectra(mz_tolerance=config_obj.mz_tol_to_remove_blank, rt_tolerance=config_obj.rt_tol_to_remove_blank)

    # ------------------------------------------------
    # Add external compound data to reference spectra
    # ------------------------------------------------
    if config_obj.compound_table_paths:
        add_classyfire_class(config_obj.compound_table_paths)

    if config_obj.metacyc_cmpd_dat_path:
        read_meta.convert_metacyc_compounds_dat_to_h5(config_obj.metacyc_cmpd_dat_path, output_path='./metacyc.h5',
                                                      parameters_to_open_file=dict(encoding='utf8', errors='replace'))
    if config_obj.metacyc_pathway_dat_path:
        read_meta.convert_metacyc_pathways_dat_to_h5(config_obj.metacyc_pathway_dat_path, output_path='./metacyc.h5',
                                                     parameters_to_open_file=dict(encoding='utf8', errors='replace'))
    if config_obj.metacyc_cmpd_dat_path and config_obj.metacyc_pathway_dat_path:
        read_meta.assign_pathway_id_to_compound_in_h5('./metacyc.h5')
    
    add_metacyc_compound_info()
    
    # ------------------------------------
    # External compound file for filtering
    # ------------------------------------
    logger.debug('Read external compound file for filtering')
    for _path in config_obj.list_path_compound_dat_for_filter:
        logger.debug(f"filename: {os.path.basename(_path)}")
        read_meta.read_metacyc_compounds_dat(_path, output_path='./metacyc_for_filter.h5',
                                             parameters_to_open_file=dict(encoding='utf8', errors='replace'))

    report.info(f"\nfiles NOT to be filtered {str(config_obj.list_filename_avoid_filter)}\n")
    filter_reference_spectra(config_obj, report)

    # ------------------------------------
    # Extract top X peak rich spectra
    # ------------------------------------
    if config_obj.num_top_x_peak_rich:
        for _filename in config_obj.list_sample_filename:
            extract_top_x_prak_rich(_filename, config_obj.num_top_x_peak_rich)
    
    serialize_filtered_spectra()

    # -------------------------------
    # Calculate spectral similarity
    # -------------------------------
    calculate_similarity_score(config_obj.mz_tol)
    clustering_based_on_inchikey()
    add_cluster_id()

    # -------
    # Output
    # -------
    write_edge_info(edge_info_path, config_obj.score_threshold_to_output, config_obj.mz_tol)
    write_cluster_attribute(cluster_attribute_path, config_obj.ref_split_category)
