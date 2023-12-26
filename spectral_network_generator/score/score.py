import itertools
import logging
from logging import DEBUG, Formatter, getLogger, StreamHandler
from matchms import calculate_scores
import numpy as np
import numpy.lib.recfunctions as rfn
import os
import pickle
from . import CosineGreedyWithMatchedIndex, ModifiedCosineWithMatchedIndex
from my_parser.matchms_spectrum_parser import (get_grouped_spectra_dirs,
                                               get_serialized_spectra_paths)
from my_parser.spectrum_metadata_parser import get_npy_metadata_paths
from utils.spectrum_processing import set_intensity_in_log1p


LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


def _calculate_similarity_score_with_cosine_greedy(references, queries, tolerance, mz_power=0, intensity_power=1, is_symmetric=False, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    return calculate_scores(references=references, queries=queries,
                            similarity_function=CosineGreedyWithMatchedIndex(
                                tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power),
                            is_symmetric=is_symmetric)


def _calculate_similarity_score_with_modified_cosine_greedy(references, queries, tolerance, mz_power=0, intensity_power=1, is_symmetric=False):    
    return calculate_scores(references=references, queries=queries,
                            similarity_function=ModifiedCosineWithMatchedIndex(
                                tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power),
                            is_symmetric=is_symmetric)


def _calculate_similarity_score_for_grouped_spectra(
        dir_output, pickle_path_a, pickle_path_b, matching_mode, tolerance, intensity_convert_mode,
        metadata_path_a, metadata_path_b, index=0, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)

    # Define intensity_power ---------------------------------------------------------
    intensity_convert_message = {0: 'do nothing', 2: 'log(1 + x)', 3: 'square root'}

    if intensity_convert_mode == 3:
        intensity_power = 0.5
    elif intensity_convert_mode == 0 or intensity_convert_mode == 2:
        intensity_power = 1
    else:
        intensity_power = 1
        intensity_convert_message[intensity_convert_mode] = 'do nothing'
    # --------------------------------------------------------------------------------

    # Make output folder -------------
    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)
    # --------------------------------

    filename_a = os.path.splitext(os.path.basename(pickle_path_a))[0]
    filename_b = os.path.splitext(os.path.basename(pickle_path_b))[0]
    folder_name_a = os.path.basename(os.path.dirname(pickle_path_a))
    folder_name_b = os.path.basename(os.path.dirname(pickle_path_b))

    is_symmetric = False
    if pickle_path_a == pickle_path_b:
        is_symmetric = True

    # Load spectra from pickle file -----------
    with open(pickle_path_a, 'rb') as f:
        spectra_a = pickle.load(f)

    if is_symmetric:
        spectra_b = spectra_a
    else:
        with open(pickle_path_b, 'rb') as f:
            spectra_b = pickle.load(f)
    # -----------------------------------------

    # Convert intensity ---------------------------------------------
    if intensity_convert_mode == 2:
        spectra_a = list(map(set_intensity_in_log1p, spectra_a))

        if is_symmetric:
            spectra_b = spectra_a
        else:
            spectra_b = list(map(set_intensity_in_log1p, spectra_b))
    # ---------------------------------------------------------------

    # Calculate similarity score ---------------------------------------------------------------------------
    if matching_mode == 1:
        logger.debug(f'Calculate spectral similarity using MatchMS CosineGreedy.\n'
                     f'{folder_name_a} {filename_a}   VS   {folder_name_b} {filename_b},'
                     f'intensity conversion: {intensity_convert_message[intensity_convert_mode]}')
        scores = _calculate_similarity_score_with_cosine_greedy(
            spectra_a, spectra_b, tolerance, intensity_power=intensity_power, is_symmetric=is_symmetric)
    elif matching_mode == 2:
        logger.debug(f'Calculate spectral similarity using MatchMS ModifiedCosine.\n'
                     f'{folder_name_a} {filename_a}   VS   {folder_name_b} {filename_b},'
                     f'intensity conversion: {intensity_convert_message[intensity_convert_mode]}')
        scores = _calculate_similarity_score_with_modified_cosine_greedy(
            spectra_a, spectra_b, tolerance, intensity_power=intensity_power, is_symmetric=is_symmetric)
    # ------------------------------------------------------------------------------------------------------

    # Get spectral metadata -----------------------------------------------------------------------
    spectrum_index_arr_a = []
    spectrum_index_arr_b = []
    for _s in spectra_a:
        spectrum_index_arr_a.append(_s.get('index'))
    for _s in spectra_b:
        spectrum_index_arr_b.append(_s.get('index'))

    # Load metadata
    metadata_arr_a = np.load(metadata_path_a, allow_pickle=True)
    metadata_arr_b = np.load(metadata_path_b, allow_pickle=True)

    # Extract metadata
    metadata_arr_a = metadata_arr_a[np.isin(metadata_arr_a['index'], spectrum_index_arr_a)]
    metadata_arr_b = metadata_arr_b[np.isin(metadata_arr_b['index'], spectrum_index_arr_b)]

    metadata_arr_a = metadata_arr_a[['keyword', 'index', 'cluster_id', 'cluster_name', 'global_accession']]
    metadata_arr_b = metadata_arr_b[['keyword', 'index', 'cluster_id', 'cluster_name', 'global_accession']]
    # ---------------------------------------------------------------------------------------------

    # Construct score structure ----------------------------------------------------------------
    # Construct metadata structure
    score_data = list(itertools.product(metadata_arr_a, metadata_arr_b))
    # Combine scores and metadata
    score_data = list(map(lambda x: (
                                        x[0][0][0], x[0][1][0],  # keyword_a, keyword_b
                                        x[0][0][1], x[0][1][1],  # index_a, index_b
                                        x[0][0][2], x[0][1][2],  # cluster_id_a, cluster_id_b
                                        x[0][0][3], x[0][1][3],  # cluster_name_a, cluster_name_b
                                        x[0][0][4], x[0][1][4],  # global_accession_a, global_accession_b
                                        x[1][0], x[1][1],  # score, matches
                                        x[1][2], x[1][3]  # matched_peak_idx_a, matched_peak_idx_b
                                    ),
                          zip(score_data, scores.scores.flatten())))

    score_arr = np.array(score_data,
                         dtype=[('keyword_a', 'O'), ('keyword_b', 'O'),
                                ('index_a', 'u8'), ('index_b', 'u8'),
                                ('cluster_id_a', 'u8'), ('cluster_id_b', 'u8'),
                                ('cluster_name_a', 'O'), ('cluster_name_b', 'O'),
                                ('global_accession_a', 'O'), ('global_accession_b', 'O'),
                                ('score', 'f8'), ('matches', 'u2'),
                                ('matched_peak_idx_a', 'O'), ('matched_peak_idx_b', 'O')])
    
    if is_symmetric:
        mask = score_arr['index_a'] < score_arr['index_b']
        score_arr = score_arr[mask]
    index_arr = np.arange(index, index + score_arr.shape[0])
    score_arr = rfn.append_fields(score_arr, 'index', index_arr, usemask=False)
    # ------------------------------------------------------------------------------------------

    # Export score as .npy file -----------------------------------------------
    pickle_dir_name_a = os.path.basename(os.path.dirname(pickle_path_a))
    pickle_dir_name_b = os.path.basename(os.path.dirname(pickle_path_b))
    score_path = os.path.join(dir_output, f'{pickle_dir_name_a}__{filename_a}_vs_{pickle_dir_name_b}__{filename_b}.npy')
    with open(score_path, 'wb') as f:
        np.save(f, score_arr)
        f.flush()
    # -------------------------------------------------------------------------

    index += score_arr.shape[0]

    return index


def calculate_similarity_score_for_grouped_spectra(sample_spectra_parent_dir,
                                                   ref_spectra_parent_dir,
                                                   output_parent_dir,
                                                   matching_mode,
                                                   tolerance,
                                                   intensity_convert_mode,
                                                   sample_metadata_dir,
                                                   ref_metadata_dir,
                                                   calculate_inner_sample=True,
                                                   calculate_inter_sample=True,
                                                   calculate_inter_sample_and_ref=True,
                                                   calculate_inner_ref=True,
                                                   calculate_inter_ref=True,
                                                   _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    # Get metadata paths.
    sample_metadata_paths = get_npy_metadata_paths(sample_metadata_dir)
    ref_metadata_paths = get_npy_metadata_paths(ref_metadata_dir)

    # Get spectra folder paths for each sample dataset.
    sample_spectra_dirs = get_grouped_spectra_dirs(sample_spectra_parent_dir)

    # Get spectra folder paths for each reference dataset.
    ref_spectra_dirs = get_grouped_spectra_dirs(ref_spectra_parent_dir)

    spectra_combinations = []
    # Add inner sample combination ---------------------------------------------------------------------
    if calculate_inner_sample:
        logger.info('Calculate inner sample scores')
        for spectra_dir, dataset_idx in sample_spectra_dirs:
            metadata_path = sample_metadata_paths[dataset_idx][0]
            paths = get_serialized_spectra_paths(spectra_dir)
            spectra_dir_name = os.path.basename(spectra_dir)
            output_dir_name = f'{spectra_dir_name}_vs_{spectra_dir_name}'
            output_dir = os.path.join(output_parent_dir, output_dir_name)

            for (path_a, _, _), (path_b, _, _) in itertools.combinations_with_replacement(paths, 2):
                spectra_combinations.append((output_dir, path_a, path_b, metadata_path, metadata_path))
    # --------------------------------------------------------------------------------------------------
    # Add inter sample combination ---------------------------------------------------------------------
    if calculate_inter_sample:
        logger.info('Calculate inter sample scores')
        for (spectra_dir_a, dataset_idx_a), (spectra_dir_b, dataset_idx_b)\
                in list(itertools.combinations(sample_spectra_dirs, 2)):
            metadata_path_a = sample_metadata_paths[dataset_idx_a][0]
            metadata_path_b = sample_metadata_paths[dataset_idx_b][0]
            paths_a = get_serialized_spectra_paths(spectra_dir_a)
            paths_b = get_serialized_spectra_paths(spectra_dir_b)
            spectra_dir_name_a = os.path.basename(spectra_dir_a)
            spectra_dir_name_b = os.path.basename(spectra_dir_b)
            output_dir_name = f'{spectra_dir_name_a}_vs_{spectra_dir_name_b}'
            output_dir = os.path.join(output_parent_dir, output_dir_name)

            for (path_a, _, _), (path_b, _, _) in list(itertools.product(paths_a, paths_b)):
                spectra_combinations.append((output_dir, path_a, path_b, metadata_path_a, metadata_path_b))
    # --------------------------------------------------------------------------------------------------
    # Add inter sample and reference combination -------------------------------------------------------
    if calculate_inter_sample_and_ref:
        logger.info('Calculate inter sample and reference scores')
        for (spectra_dir_a, dataset_idx_a), (spectra_dir_b, dataset_idx_b) \
                in list(itertools.product(sample_spectra_dirs, ref_spectra_dirs)):
            metadata_path_a = sample_metadata_paths[dataset_idx_a][0]
            metadata_path_b = ref_metadata_paths[dataset_idx_b][0]
            paths_a = get_serialized_spectra_paths(spectra_dir_a)
            paths_b = get_serialized_spectra_paths(spectra_dir_b)
            spectra_dir_name_a = os.path.basename(spectra_dir_a)
            spectra_dir_name_b = os.path.basename(spectra_dir_b)
            output_dir_name = f'{spectra_dir_name_a}_vs_{spectra_dir_name_b}'
            output_dir = os.path.join(output_parent_dir, output_dir_name)

            for (path_a, _, _), (path_b, _, _) in list(itertools.product(paths_a, paths_b)):
                spectra_combinations.append((output_dir, path_a, path_b, metadata_path_a, metadata_path_b))
    # --------------------------------------------------------------------------------------------------
    # Add inner reference combination ------------------------------------------------------------------
    if calculate_inner_ref:
        logger.info('Calculate inner reference scores')
        for spectra_dir, dataset_idx in ref_spectra_dirs:
            metadata_path = ref_metadata_paths[dataset_idx][0]
            paths = get_serialized_spectra_paths(spectra_dir)
            spectra_dir_name = os.path.basename(spectra_dir)
            output_dir_name = f'{spectra_dir_name}_vs_{spectra_dir_name}'
            output_dir = os.path.join(output_parent_dir, output_dir_name)

            for (path_a, _, _), (path_b, _, _) in itertools.combinations_with_replacement(paths, 2):
                spectra_combinations.append((output_dir, path_a, path_b, metadata_path, metadata_path))
    # --------------------------------------------------------------------------------------------------
    # Add inter reference combination ------------------------------------------------------------------
    if calculate_inter_ref:
        logger.info('Calculate inter reference scores')
        for (spectra_dir_a, dataset_idx_a), (spectra_dir_b, dataset_idx_b) \
                in list(itertools.combinations(ref_spectra_dirs, 2)):
            metadata_path_a = ref_metadata_paths[dataset_idx_a][0]
            metadata_path_b = ref_metadata_paths[dataset_idx_b][0]
            paths_a = get_serialized_spectra_paths(spectra_dir_a)
            paths_b = get_serialized_spectra_paths(spectra_dir_b)
            spectra_dir_name_a = os.path.basename(spectra_dir_a)
            spectra_dir_name_b = os.path.basename(spectra_dir_b)
            output_dir_name = f'{spectra_dir_name_a}_vs_{spectra_dir_name_b}'
            output_dir = os.path.join(output_parent_dir, output_dir_name)

            for (path_a, _, _), (path_b, _, _) in list(itertools.product(paths_a, paths_b)):
                spectra_combinations.append((output_dir, path_a, path_b, metadata_path_a, metadata_path_b))
    # --------------------------------------------------------------------------------------------------

    # Calculate scores
    index = 0
    for output_dir, path_a, path_b, metadata_path_a, metadata_path_b in spectra_combinations:
        index = _calculate_similarity_score_for_grouped_spectra(
            dir_output=output_dir,
            pickle_path_a=path_a,
            pickle_path_b=path_b,
            matching_mode=matching_mode,
            tolerance=tolerance,
            intensity_convert_mode=intensity_convert_mode,
            metadata_path_a=metadata_path_a,
            metadata_path_b=metadata_path_b,
            index=index
        )
