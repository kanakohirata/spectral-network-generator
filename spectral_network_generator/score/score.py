from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from glob import glob
import h5py
import itertools
import logging
from logging import DEBUG, Formatter, getLogger, StreamHandler
from matchms import calculate_scores
import numpy as np
import numpy.lib.recfunctions as rfn
import os
import pickle
import re
from . import CosineGreedyWithMatchedIndex, ModifiedCosineWithMatchedIndex
from my_parser.matchms_spectrum_parser import (get_grouped_spectra_dirs,
                                               get_ref_spectra_paths,
                                               get_sample_spectra_paths,
                                               get_serialized_spectra_paths)
from my_parser.spectrum_metadata_parser import get_npy_metadata_paths
from my_parser.score_parser import initialize_score_hdf5, iter_clustered_score_array, iter_score_array
from utils.spectrum_processing import set_intensity_in_log1p


LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


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


def calculate_similarity_score(matching_mode, tolerance, intensity_convert_mode, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    initialize_score_hdf5()

    intensity_convert_message = {0: 'do nothing', 2: 'log(1 + x)', 3: 'square root'}

    if intensity_convert_mode == 3:
        intensity_power = 0.5
    elif intensity_convert_mode == 0 or intensity_convert_mode == 2:
        intensity_power = 1
    else:
        intensity_power = 1
        intensity_convert_message[intensity_convert_mode] = 'do nothing'

    pickle_paths = glob('./serialized_spectra/filtered/*.pickle')
    path_vs_index_list = [(path, int(os.path.basename(path).split('-')[0])) for path in pickle_paths]
    path_vs_index_list.sort(key=lambda x: x[1])
    pickle_paths = [x[0] for x in path_vs_index_list]

    pickle_path_combinations = itertools.combinations_with_replacement(pickle_paths, 2)
    index = 0
    with h5py.File('./spectrum_metadata.h5', 'a') as h5_metadata:
        for pickle_path_a, pickle_path_b in pickle_path_combinations:
            filename_a = os.path.splitext(os.path.basename(pickle_path_a))[0]
            filename_b = os.path.splitext(os.path.basename(pickle_path_b))[0]

            is_symmetric = False
            if pickle_path_a == pickle_path_b:
                is_symmetric = True
            
            with open(pickle_path_a, 'rb') as f:
                spectra_a = pickle.load(f)

            if is_symmetric:
                spectra_b = spectra_a
            else:
                with open(pickle_path_b, 'rb') as f:
                    spectra_b = pickle.load(f)
            
            if intensity_convert_mode == 2:
                spectra_a = list(map(set_intensity_in_log1p, spectra_a))

                if is_symmetric:
                    spectra_b = spectra_a
                else:
                    spectra_b = list(map(set_intensity_in_log1p, spectra_b))

            logger.debug(f'Calculate spectral similarity using MatchMS.\n'
                         f'{filename_a} vs {filename_b}, intensity conversion: {intensity_convert_message[intensity_convert_mode]}')

            if matching_mode == 1:
                scores = _calculate_similarity_score_with_cosine_greedy(
                    spectra_a, spectra_b, tolerance, intensity_power=intensity_power, is_symmetric=is_symmetric)
            elif matching_mode == 2:
                scores = _calculate_similarity_score_with_modified_cosine_greedy(
                    spectra_a, spectra_b, tolerance, intensity_power=intensity_power, is_symmetric=is_symmetric)

            # Get spectral metadata.
            spectrum_idx_start_a = int(re.findall(r'\d+', filename_a)[0])
            spectrum_idx_end_a = int(re.findall(r'\d+', filename_a)[1])
            spectrum_idx_start_b = int(re.findall(r'\d+', filename_b)[0])
            spectrum_idx_end_b = int(re.findall(r'\d+', filename_b)[1])

            metadata_arr_a = h5_metadata['filtered/metadata'][spectrum_idx_start_a:spectrum_idx_end_a + 1]
            metadata_arr_b = h5_metadata['filtered/metadata'][spectrum_idx_start_b:spectrum_idx_end_b + 1]

            metadata_arr_a = metadata_arr_a[['index', 'global_accession', 'tag', 'inchikey']]
            metadata_arr_b = metadata_arr_b[['index', 'global_accession', 'tag', 'inchikey']]

            score_data = list(itertools.product(metadata_arr_a, metadata_arr_b))

            score_data = list(map(lambda x: (x[0][0][0], x[0][1][0],
                                             x[0][0][1], x[0][1][1],
                                             x[0][0][2], x[0][1][2],
                                             x[0][0][3], x[0][1][3],
                                             x[1][0], x[1][1]),
                                  zip(score_data, scores.scores.flatten())))
            
            score_arr = np.array(score_data,
                                 dtype=[('index_a', 'u8'), ('index_b', 'u8'),
                                        ('global_accession_a', 'O'), ('global_accession_b', 'O'),
                                        ('tag_a', 'S6'), ('tag_b', 'S6'),
                                        ('inchikey_a', 'S27'), ('inchikey_b', 'S27'),
                                        ('score', 'f8'), ('matches', 'u2')])
            
            max_len_of_global_accession_a = max(len(x) for x in score_arr['global_accession_a'])
            max_len_of_global_accession_b = max(len(x) for x in score_arr['global_accession_b'])

            score_arr = score_arr.astype([('index_a', 'u8'), ('index_b', 'u8'),
                                          ('global_accession_a', f'S{max_len_of_global_accession_a}'),
                                          ('global_accession_b', f'S{max_len_of_global_accession_b}'),
                                          ('tag_a', f'S6'), ('tag_b', f'S6'),
                                          ('inchikey_a', f'S27'), ('inchikey_b', f'S27'),
                                          ('score', 'f8'), ('matches', 'u2')])

            mask = score_arr['index_a'] < score_arr['index_b']
            score_arr = score_arr[mask]
            index_arr = np.arange(index, index + score_arr.shape[0])
            score_arr = rfn.append_fields(score_arr, 'index', index_arr, usemask=False)

            score_path = f'./scores/{filename_a}_vs_{filename_b}.npy'
            with open(score_path, 'wb') as f:
                np.save(f, score_arr)

            index += score_arr.shape[0]


def _calculate_similarity_score_for_grouped_spectra_old(
        dir_output, pickle_path_a, pickle_path_b, matching_mode, tolerance, intensity_convert_mode,
        metadata_key_a, metadata_key_b, index=0, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER
    
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
    logger.debug(f'Calculate spectral similarity using MatchMS.\n'
                    f'{filename_a} vs {filename_b}, intensity conversion: {intensity_convert_message[intensity_convert_mode]}')

    if matching_mode == 1:
        scores = _calculate_similarity_score_with_cosine_greedy(
            spectra_a, spectra_b, tolerance, intensity_power=intensity_power, is_symmetric=is_symmetric)
    elif matching_mode == 2:
        scores = _calculate_similarity_score_with_modified_cosine_greedy(
            spectra_a, spectra_b, tolerance, intensity_power=intensity_power, is_symmetric=is_symmetric)
    # ------------------------------------------------------------------------------------------------------

    # Get spectral metadata -----------------------------------------------------------------------
    spectrum_idx_start_a = int(re.findall(r'\d+', filename_a)[0])
    spectrum_idx_end_a = int(re.findall(r'\d+', filename_a)[1])
    spectrum_idx_start_b = int(re.findall(r'\d+', filename_b)[0])
    spectrum_idx_end_b = int(re.findall(r'\d+', filename_b)[1])

    # Read metadata
    with h5py.File('./spectrum_metadata.h5', 'r') as h5_metadata:
        metadata_arr_a = h5_metadata[metadata_key_a][spectrum_idx_start_a:spectrum_idx_end_a + 1]
        metadata_arr_b = h5_metadata[metadata_key_b][spectrum_idx_start_b:spectrum_idx_end_b + 1]

    metadata_arr_a = metadata_arr_a[['index', 'global_accession', 'tag', 'inchikey']]
    metadata_arr_b = metadata_arr_b[['index', 'global_accession', 'tag', 'inchikey']]
    # ---------------------------------------------------------------------------------------------

    # Construct score structure ----------------------------------------------------------------
    # Construct metadata structure
    score_data = list(itertools.product(metadata_arr_a, metadata_arr_b))
    # Combine scores and metadata
    score_data = list(map(lambda x: (x[0][0][0], x[0][1][0],
                                     x[0][0][1], x[0][1][1],
                                     x[0][0][2], x[0][1][2],
                                     x[0][0][3], x[0][1][3],
                                     x[1][0], x[1][1]),
                          zip(score_data, scores.scores.flatten())))
    
    score_arr = np.array(score_data,
                         dtype=[('index_a', 'u8'), ('index_b', 'u8'),
                                ('global_accession_a', 'O'), ('global_accession_b', 'O'),
                                ('tag_a', 'S6'), ('tag_b', 'S6'),
                                ('inchikey_a', 'S27'), ('inchikey_b', 'S27'),
                                ('score', 'f8'), ('matches', 'u2')])
    
    max_len_of_global_accession_a = max(len(x) for x in score_arr['global_accession_a'])
    max_len_of_global_accession_b = max(len(x) for x in score_arr['global_accession_b'])

    score_arr = score_arr.astype([('index_a', 'u8'), ('index_b', 'u8'),
                                  ('global_accession_a', f'S{max_len_of_global_accession_a}'),
                                  ('global_accession_b', f'S{max_len_of_global_accession_b}'),
                                  ('tag_a', f'S6'), ('tag_b', f'S6'),
                                  ('inchikey_a', f'S27'), ('inchikey_b', f'S27'),
                                  ('score', 'f8'), ('matches', 'u2')])

    mask = score_arr['index_a'] < score_arr['index_b']
    score_arr = score_arr[mask]
    index_arr = np.arange(index, index + score_arr.shape[0])
    score_arr = rfn.append_fields(score_arr, 'index', index_arr, usemask=False)
    # ------------------------------------------------------------------------------------------

    # Export score as .npy file -----------------------------------------------
    score_path = os.path.join(dir_output, f'{filename_a}_vs_{filename_b}.npy')
    with open(score_path, 'wb') as f:
        np.save(f, score_arr)
    # -------------------------------------------------------------------------

    index += score_arr.shape[0]

    return index


def calculate_similarity_score_for_grouped_spectra_old(matching_mode, tolerance, intensity_convert_mode, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    sample_spectra_paths = get_sample_spectra_paths()
    ref_dataset_keyword_vs_spectra_paths = get_ref_spectra_paths()

    index = 0

    # Calculate scores of sample vs sample ------------------------------------------------------------------------
    logger.info('Calculate scores of sample vs sample')
    sample_score_dir = './scores/grouped_scores/sample'
    for (pickle_path_a, _, _), (pickle_path_b, _, _) in itertools.combinations_with_replacement(sample_spectra_paths, 2):
        index = _calculate_similarity_score_for_grouped_spectra(
            sample_score_dir, pickle_path_a, pickle_path_b, matching_mode, tolerance, intensity_convert_mode,
            'grouped/sample', 'grouped/sample', index
        )
    # --------------------------------------------------------------------------------------------------------------

    # Calculate scores of sample vs reference ---------------------------------------------------------------------
    for dataset_keyword, ref_spectra_paths in ref_dataset_keyword_vs_spectra_paths.items():
        logger.info(f'Calculate scores of sample vs {dataset_keyword}')
        sample_vs_ref_score_dir = os.path.join(sample_score_dir, dataset_keyword)
        for (pickle_path_a, _, _), (pickle_path_b, _, _) in itertools.product(sample_spectra_paths, ref_spectra_paths):
            index = _calculate_similarity_score_for_grouped_spectra(
                sample_vs_ref_score_dir, pickle_path_a, pickle_path_b, matching_mode, tolerance, intensity_convert_mode,
                'grouped/sample', f'grouped/{dataset_keyword}', index
            )
    # --------------------------------------------------------------------------------------------------------------

    # Calculate scores of reference vs reference ------------------------------------------------------------------
    for dataset_keyword, ref_spectra_paths in ref_dataset_keyword_vs_spectra_paths.items():
        logger.info(f'Calculate scores of {dataset_keyword} vs {dataset_keyword}')
        ref_score_dir = os.path.join('./scores/grouped_scores', dataset_keyword)
        for (pickle_path_a, _, _), (pickle_path_b, _, _) in itertools.combinations_with_replacement(ref_spectra_paths, 2):
            index = _calculate_similarity_score_for_grouped_spectra(
                ref_score_dir, pickle_path_a, pickle_path_b, matching_mode, tolerance, intensity_convert_mode,
                f'grouped/{dataset_keyword}', f'grouped/{dataset_keyword}', index
            )
    # --------------------------------------------------------------------------------------------------------------


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


def _cluster_score_core(score_arr, cluster_ids, clustered_score_dir='./scores/clustered_scores'):
    # Get cluster index
    cluster_index_arr_a = np.array((list(map(lambda x: cluster_ids.index(x), score_arr['cluster_id_a']))))
    cluster_index_arr_b = np.array(list(map(lambda x: cluster_ids.index(x), score_arr['cluster_id_b'])))
    
    # Get max length of cluster id
    max_length_of_cluster_id = max(len(x) for x in score_arr['cluster_id_a'])
    max_length_of_cluster_id = max(max_length_of_cluster_id, max(len(x) for x in score_arr['cluster_id_b']))

    # Create un-clustered score array (score_arr + cluster_index_arr_a and cluster_index_arr_b)
    dtype = [('index', 'u8'),
             ('cluster_index_a', 'u8'), ('cluster_index_b', 'u8'),
             ('index_a', 'u8'), ('index_b', 'u8'),
             ('score', 'f2'), ('matches', 'u2'),
             ('cluster_id_a', f'S{max_length_of_cluster_id}'),
             ('cluster_id_b', f'S{max_length_of_cluster_id}')]
    unclustered_score_arr = np.zeros(cluster_index_arr_a.size, dtype=dtype)
    
    unclustered_score_arr['cluster_index_a'] = cluster_index_arr_a
    unclustered_score_arr['cluster_index_b']  = cluster_index_arr_b
    unclustered_score_arr[['index_a', 'index_b', 'score', 'matches', 'cluster_id_a', 'cluster_id_b']]\
        = score_arr[['index_a', 'index_b', 'score', 'matches', 'cluster_id_a', 'cluster_id_b']]
    del score_arr

    # Rearrange unclustered_score_arr so that unclustered_score_arr['cluster_index_a'] < unclustered_score_arr['cluster_index_b']
    mask_to_replace= cluster_index_arr_a > cluster_index_arr_b
    if np.any(mask_to_replace):
        _arr_to_replace = np.zeros(cluster_index_arr_a[mask_to_replace].shape[0], dtype=dtype)
        _arr_to_replace['cluster_index_a'] = unclustered_score_arr[mask_to_replace]['cluster_index_b']
        _arr_to_replace['cluster_index_b'] = unclustered_score_arr[mask_to_replace]['cluster_index_a']
        _arr_to_replace['index_a'] = unclustered_score_arr[mask_to_replace]['index_b']
        _arr_to_replace['index_b'] = unclustered_score_arr[mask_to_replace]['index_a']
        _arr_to_replace['score'] = unclustered_score_arr[mask_to_replace]['score']
        _arr_to_replace['matches'] = unclustered_score_arr[mask_to_replace]['matches']
        _arr_to_replace['cluster_id_a'] = unclustered_score_arr[mask_to_replace]['cluster_id_b']
        _arr_to_replace['cluster_id_b'] = unclustered_score_arr[mask_to_replace]['cluster_id_a']

        unclustered_score_arr[mask_to_replace] = _arr_to_replace
        del _arr_to_replace

    # Sort by cluster_index_a and cluster_index_b
    unclustered_score_arr = unclustered_score_arr[np.argsort(unclustered_score_arr, order=['cluster_index_a', 'cluster_index_b'])]

    for clustered_score_arr, clustered_score_path, clustered_score_idx_start in iter_clustered_score_array(dir_path=clustered_score_dir):
        if clustered_score_arr is None:
            continue
        
        # Extract scores where unclustered_score_arr[['cluster_index_a', 'cluster_index_b']]
        # is in clustered_score_arr[['cluster_index_a', 'cluster_index_b']]
        mask_is_in_clustered = np.isin(unclustered_score_arr[['cluster_index_a', 'cluster_index_b']],
                                       clustered_score_arr[['cluster_index_a', 'cluster_index_b']])
        if not np.any(mask_is_in_clustered):
            continue
        
        unclustered_score_arr_temp = unclustered_score_arr[mask_is_in_clustered]

        # Extract scores where clustered_score_arr[['cluster_index_a', 'cluster_index_b']]
        # is in unclustered_score_arr[['cluster_index_a', 'cluster_index_b']]
        mask_is_in_unclustered = np.isin(clustered_score_arr[['cluster_index_a', 'cluster_index_b']],
                                         unclustered_score_arr_temp[['cluster_index_a', 'cluster_index_b']])
        clustered_score_arr_temp = clustered_score_arr[mask_is_in_unclustered]

        _test = clustered_score_arr_temp[['cluster_index_a', 'cluster_index_b']] == unclustered_score_arr_temp[['cluster_index_a', 'cluster_index_b']]
        if not np.all(_test):
            raise ValueError('Order of "cluster_index_a" and "cluster_index_b" must be the same between arrays.')
        
        unclustered_score_arr_temp['index'] = clustered_score_arr_temp['index']

        # Match the order of fields in clustered_score_arr_temp and unclustered_score_arr_temp.
        fields = [x for x in clustered_score_arr_temp.dtype.names]
        unclustered_score_arr_temp = unclustered_score_arr_temp[fields]

        # Get bool array indicating whether clustered_score_arr_temp['score'] < unclustered_score_arr_temp['score']
        mask_score = (clustered_score_arr_temp['score'] < unclustered_score_arr_temp['score'])\
                     | ((clustered_score_arr_temp['score'] == unclustered_score_arr_temp['score'])
                        & (clustered_score_arr_temp['matches'] <= unclustered_score_arr_temp['matches']))
        
        # Replace clustered_score_arr_temp where clustered_score_arr_temp['score'] < unclustered_score_arr_temp['score']
        # with unclustered_score_arr_temp
        clustered_score_arr_temp[mask_score] = unclustered_score_arr_temp[mask_score]
        
        # Update clustered_score_arr
        clustered_score_arr[mask_is_in_unclustered] = clustered_score_arr_temp

        with open(clustered_score_path, 'wb') as clustered_score_file:
            np.save(clustered_score_file, clustered_score_arr)
        
        print(1)


def _cluster_sample_vs_sample_score(score_arr, cluster_ids):    
    # Add 'cluster_id_a' and 'cluster_id_b' fields
    score_arr = rfn.append_fields(score_arr, 'cluster_id_a', score_arr['global_accession_a'], usemask=False)
    score_arr = rfn.append_fields(score_arr, 'cluster_id_b', score_arr['global_accession_b'], usemask=False)

    _cluster_score_core(score_arr, cluster_ids)


def _cluster_sample_vs_ref_score(score_arr, cluster_ids):
    # Add 'cluster_id_a' and 'cluster_id_b' fields
    cluster_id_arr_b = np.where(score_arr['inchikey_b'] == b'',
                                score_arr['global_accession_b'],
                                score_arr['inchikey_b'])
    score_arr = rfn.append_fields(score_arr, 'cluster_id_a', score_arr['global_accession_a'], usemask=False)
    score_arr = rfn.append_fields(score_arr, 'cluster_id_b', cluster_id_arr_b, usemask=False)
    
    # Sort by similarity score and number of marched peaks
    score_arr = score_arr[np.argsort(score_arr, order=['score', 'matches'])[::-1]]
    # Extract scores with unique combination of cluster_id_a and cluster_id_b
    temp_arr = score_arr[['cluster_id_a', 'cluster_id_b']]
    unique_indexes = np.unique(temp_arr, axis=0, return_index=True)[1]
    score_arr = score_arr[unique_indexes]
    
    _cluster_score_core(score_arr, cluster_ids)


def _cluster_ref_vs_sample_score(score_arr, cluster_ids):
    # Add 'cluster_id_a' and 'cluster_id_b' fields
    cluster_id_arr_a = np.where(score_arr['inchikey_a'] == b'',
                                score_arr['global_accession_a'],
                                score_arr['inchikey_a'])
    score_arr = rfn.append_fields(score_arr, 'cluster_id_a', cluster_id_arr_a, usemask=False)
    score_arr = rfn.append_fields(score_arr, 'cluster_id_b', score_arr['global_accession_b'], usemask=False)

    # Sort by similarity score and number of marched peaks
    score_arr = score_arr[np.argsort(score_arr, order=['score', 'matches'])[::-1]]
    # Extract scores with unique combination of cluster_id_a and cluster_id_b
    temp_arr = np.array(score_arr[['cluster_id_a', 'cluster_id_b']].tolist())
    unique_indexes = np.unique(temp_arr, axis=0, return_index=True)[1]
    score_arr = score_arr[unique_indexes]

    _cluster_score_core(score_arr, cluster_ids)


def _cluster_ref_vs_ref_score(score_arr, cluster_ids):
    # Add 'cluster_id_a' and 'cluster_id_b' fields
    cluster_id_arr_a = np.where(score_arr['inchikey_a'] == b'',
                                score_arr['global_accession_a'],
                                score_arr['inchikey_a'])
    cluster_id_arr_b = np.where(score_arr['inchikey_b'] == b'',
                                score_arr['global_accession_b'],
                                score_arr['inchikey_b'])
    score_arr = rfn.append_fields(score_arr, 'cluster_id_a', cluster_id_arr_a, usemask=False)
    score_arr = rfn.append_fields(score_arr, 'cluster_id_b', cluster_id_arr_b, usemask=False)

    # Sort by similarity score and number of matched peaks
    score_arr = score_arr[np.argsort(score_arr, order=['score', 'matches'])[::-1]]
    # Extract scores with unique combination of cluster_id_a and cluster_id_b
    temp_arr = np.array(score_arr[['cluster_id_a', 'cluster_id_b']].tolist())
    temp_arr = np.sort(temp_arr, axis=1)
    unique_indices = np.unique(temp_arr, axis=0, return_index=True)[1]
    score_arr = score_arr[unique_indices]

    _cluster_score_core(score_arr, cluster_ids)


def clustering_based_on_inchikey(_logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    with open('./cluster_ids.pickle', 'rb') as f:
        cluster_ids = pickle.load(f)

    for arr, score_path in iter_score_array(return_index=False):
        logger.info(f'Clustering {os.path.basename(score_path)}')

        sample_vs_sample_mask = (arr['tag_a'] == b'sample') & (arr['tag_b'] == b'sample')
        sample_vs_ref_mask = (arr['tag_a'] == b'sample') & (arr['tag_b'] == b'ref')
        ref_vs_sample_mask = (arr['tag_a'] == b'ref') & (arr['tag_b'] == b'sample')
        ref_vs_ref_mask = (arr['tag_a'] == b'ref') & (arr['tag_b'] == b'ref')

        if np.any(sample_vs_sample_mask):
            _arr = arr[sample_vs_sample_mask]
            _cluster_sample_vs_sample_score(_arr, cluster_ids)

        if np.any(sample_vs_ref_mask):
            _arr = arr[sample_vs_ref_mask]
            _cluster_sample_vs_ref_score(_arr, cluster_ids)

        if np.any(ref_vs_sample_mask):
            _arr = arr[ref_vs_sample_mask]
            _cluster_ref_vs_sample_score(_arr, cluster_ids)

        if np.any(ref_vs_ref_mask):
            _arr = arr[ref_vs_ref_mask]
            _cluster_ref_vs_ref_score(_arr, cluster_ids)


def _cluster_grouped_sample_vs_sample_score(score_arr, cluster_ids, max_length_of_cluster_id):
    # Suffix '_a': sample, '_b': sample
    
    # 'cluster_id_a': 'sample|' + global_accession
    # 'cluster_id_b': 'sample|' + global_accession
    cluster_id_arr_a = np.core.defchararray.add(b'sample|', score_arr['global_accession_a'].astype(f'S{max_length_of_cluster_id}'))
    cluster_id_arr_b = np.core.defchararray.add(b'sample|', score_arr['global_accession_b'].astype(f'S{max_length_of_cluster_id}'))
    
    # Add 'cluster_id_a' and 'cluster_id_b' fields
    score_arr = rfn.append_fields(score_arr, 'cluster_id_a', cluster_id_arr_a, usemask=False)
    score_arr = rfn.append_fields(score_arr, 'cluster_id_b', cluster_id_arr_b, usemask=False)

    _cluster_score_core(score_arr, cluster_ids, './scores/grouped_and_clustered_scores/sample')


def _cluster_grouped_sample_vs_ref_score(score_arr, cluster_ids, max_length_of_cluster_id, dataset_keyword):
    # Suffix '_a': sample, '_b': reference
    
    # 'cluster_id_a': 'sample|' + global_accession
    cluster_id_arr_a = np.core.defchararray.add(b'sample|', score_arr['global_accession_a'].astype(f'S{max_length_of_cluster_id}'))
    
    # 'cluster_id_b': 'dataset_keyword|' + inchikey or global_accession
    dataset_keyword_byte = f'{dataset_keyword}|'.encode('utf8')
    cluster_id_arr_b = np.where(score_arr['inchikey_b'] == b'',
                                score_arr['global_accession_b'],
                                score_arr['inchikey_b'])
    cluster_id_arr_b = np.core.defchararray.add(dataset_keyword_byte, cluster_id_arr_b.astype(f'S{max_length_of_cluster_id}'))
    
    # Add 'cluster_id_a' and 'cluster_id_b' fields
    score_arr = rfn.append_fields(score_arr, 'cluster_id_a', cluster_id_arr_a, usemask=False)
    score_arr = rfn.append_fields(score_arr, 'cluster_id_b', cluster_id_arr_b, usemask=False)
    
    # Sort by similarity score and number of marched peaks
    score_arr = score_arr[np.argsort(score_arr, order=['score', 'matches'])[::-1]]
    # Extract scores with unique combination of cluster_id_a and cluster_id_b
    temp_arr = score_arr[['cluster_id_a', 'cluster_id_b']]
    unique_indexes = np.unique(temp_arr, axis=0, return_index=True)[1]
    score_arr = score_arr[unique_indexes]
    
    _cluster_score_core(score_arr, cluster_ids, './scores/grouped_and_clustered_scores/sample')


def _cluster_grouped_ref_vs_sample_score(score_arr, cluster_ids, max_length_of_cluster_id, dataset_keyword):
    # Suffix '_a': reference, '_b': sample

    # 'cluster_id_a': 'dataset_keyword|' + inchikey or global_accession
    dataset_keyword_byte = f'{dataset_keyword}|'.encode('utf8')
    cluster_id_arr_a = np.where(score_arr['inchikey_a'] == b'',
                                score_arr['global_accession_a'],
                                score_arr['inchikey_a'])
    cluster_id_arr_a = np.core.defchararray.add(dataset_keyword_byte, cluster_id_arr_a.astype(f'S{max_length_of_cluster_id}'))
    
    # 'cluster_id_b': 'sample|' + global_accession
    cluster_id_arr_b = np.core.defchararray.add(b'sample|', score_arr['global_accession_b'].astype(f'S{max_length_of_cluster_id}'))
    
    # Add 'cluster_id_a' and 'cluster_id_b' fields
    score_arr = rfn.append_fields(score_arr, 'cluster_id_a', cluster_id_arr_a, usemask=False)
    score_arr = rfn.append_fields(score_arr, 'cluster_id_b', cluster_id_arr_b, usemask=False)
    
    # Sort by similarity score and number of marched peaks
    score_arr = score_arr[np.argsort(score_arr, order=['score', 'matches'])[::-1]]
    # Extract scores with unique combination of cluster_id_a and cluster_id_b
    temp_arr = score_arr[['cluster_id_a', 'cluster_id_b']]
    unique_indexes = np.unique(temp_arr, axis=0, return_index=True)[1]
    score_arr = score_arr[unique_indexes]
    
    _cluster_score_core(score_arr, cluster_ids, './scores/grouped_and_clustered_scores/sample')


def _cluster_grouped_ref_vs_ref_score(score_arr, cluster_ids, max_length_of_cluster_id, dataset_keyword):
    # Suffix '_a': reference, '_b': reference

    dataset_keyword_byte = f'{dataset_keyword}|'.encode('utf8')

    # 'cluster_id_a': 'dataset_keyword|' + inchikey or global_accession
    cluster_id_arr_a = np.where(score_arr['inchikey_a'] == b'',
                                score_arr['global_accession_a'],
                                score_arr['inchikey_a'])
    cluster_id_arr_a = np.core.defchararray.add(dataset_keyword_byte, cluster_id_arr_a.astype(f'S{max_length_of_cluster_id}'))
    
    # 'cluster_id_b': 'dataset_keyword|' + inchikey or global_accession
    cluster_id_arr_b = np.where(score_arr['inchikey_b'] == b'',
                                score_arr['global_accession_b'],
                                score_arr['inchikey_b'])
    cluster_id_arr_b = np.core.defchararray.add(dataset_keyword_byte, cluster_id_arr_b.astype(f'S{max_length_of_cluster_id}'))
    
    # Add 'cluster_id_a' and 'cluster_id_b' fields
    score_arr = rfn.append_fields(score_arr, 'cluster_id_a', cluster_id_arr_a, usemask=False)
    score_arr = rfn.append_fields(score_arr, 'cluster_id_b', cluster_id_arr_b, usemask=False)
    
    # Sort by similarity score and number of matched peaks
    score_arr = score_arr[np.argsort(score_arr, order=['score', 'matches'])[::-1]]
    # Extract scores with unique combination of cluster_id_a and cluster_id_b
    temp_arr = np.array(score_arr[['cluster_id_a', 'cluster_id_b']].tolist())
    temp_arr = np.sort(temp_arr, axis=1)
    unique_indices = np.unique(temp_arr, axis=0, return_index=True)[1]
    score_arr = score_arr[unique_indices]
    
    grouped_and_clustered_scores_dir = os.path.join('./scores/grouped_and_clustered_scores', dataset_keyword)
    _cluster_score_core(score_arr, cluster_ids, grouped_and_clustered_scores_dir)


def clustering_grouped_score_based_on_inchikey(max_length_of_cluster_id, _logger=None):
    if isinstance(_logger, logging.Logger):
        logger = _logger
    else:
        logger = LOGGER

    with open('./cluster_ids.pickle', 'rb') as f:
        cluster_ids = pickle.load(f)

    # Get reference dataset keywords ------------------------
    ref_dataset_keywords = []
    with h5py.File('./spectrum_metadata.h5', 'r') as h5_metadata:
        for k in h5_metadata['grouped'].keys():
            if k != 'sample':
                ref_dataset_keywords.append(k)

    # Cluster sample vs sample scores ----------------------------------------------------------------------
    logger.info('Clustering sample vs sample scores ------------------------')
    for arr, score_path in iter_score_array(dir_path='./scores/grouped_scores/sample', return_index=False):
        logger.info(f'Clustering {os.path.basename(score_path)}')
        _cluster_grouped_sample_vs_sample_score(arr, cluster_ids, max_length_of_cluster_id)
    # ------------------------------------------------------------------------------------------------------

    # Cluster sample vs reference scores -------------------------------------------------------------------
    for dataset_keyword in ref_dataset_keywords:
        logger.info(f'Clustering sample vs {dataset_keyword} scores ------------------------')
        grouped_scores_dir = os.path.join('./scores/grouped_scores/sample', dataset_keyword)
        for arr, score_path in iter_score_array(dir_path=grouped_scores_dir, return_index=False):
            logger.info(f'Clustering {os.path.basename(score_path)}')

            sample_vs_ref_mask = (arr['tag_a'] == b'sample') & (arr['tag_b'] == b'ref')
            ref_vs_sample_mask = (arr['tag_a'] == b'ref') & (arr['tag_b'] == b'sample')

            if np.any(sample_vs_ref_mask):
                _arr = arr[sample_vs_ref_mask]
                _cluster_grouped_sample_vs_ref_score(_arr, cluster_ids, max_length_of_cluster_id, dataset_keyword)

            if np.any(ref_vs_sample_mask):
                _arr = arr[ref_vs_sample_mask]
                _cluster_grouped_ref_vs_sample_score(_arr, cluster_ids, max_length_of_cluster_id, dataset_keyword)
    # ------------------------------------------------------------------------------------------------------

    # Cluster reference vs reference scores ----------------------------------------------------------------
    for dataset_keyword in ref_dataset_keywords:
        logger.info(f'Clustering {dataset_keyword} vs {dataset_keyword} scores ------------------------')
        grouped_scores_dir = os.path.join('./scores/grouped_scores', dataset_keyword)
        for arr, score_path in iter_score_array(dir_path=grouped_scores_dir, return_index=False):
            logger.info(f'Clustering {os.path.basename(score_path)}')
            _cluster_grouped_ref_vs_ref_score(arr, cluster_ids, max_length_of_cluster_id, dataset_keyword)
    # ------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    pass
    # with h5py.File('./score.h5', 'a') as h5:
    #     if 'clustered_score' in h5.keys():
    #         del h5['clustered_score']
    #
    #     if '_temp' in h5.keys():
    #         del h5['_temp']
    #
    #     if '_ref_score' in h5.keys():
    #         del h5['_ref_score']
    #
    #     if '_clustered_ref_score' in h5.keys():
    #         del h5['_clustered_ref_score']
    #     h5.flush()
    #
    #     dset = h5['score']
    #     index_arr = dset.fields('index')[()]
    #     _, indices = np.unique(index_arr, axis=0, return_index=True)
    #     print(1)
    #
    # time_start = time.time()
    # clustering_based_on_inchikey()
    # _time = time.time() - time_start
    # print(_time)

    # with h5py.File('./score.h5', 'r') as h5:
    #     dset0 = h5['clustered_score']
    #     arr0 = dset0[(dset0['tag_a'] == b'ref') & (dset0['tag_b'] == b'ref')]
    #     if arr0.size:
    #         temp_arr = np.array(arr0[['inchikey_a', 'inchikey_b']].tolist())
    #         temp_arr = np.sort(temp_arr, axis=1)
    #         _, indices = np.unique(temp_arr, axis=0, return_index=True)
    # print(1)
