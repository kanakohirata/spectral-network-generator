import itertools
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import os
import pandas as pd
import pickle
from my_parser.spectrum_metadata_parser import get_npy_metadata_paths

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


CLUSTER_NAME_LIST_PATH = './__cluster_name_list.pickle'
PATH_OF_DIR_LIST_INNER_SAMPLE = './__clustered_score_dir_list_inner_sample.pickle'
PATH_OF_DIR_LIST_INTER_SAMPLE = './__clustered_score_dir_list_inter_sample.pickle'
PATH_OF_DIR_LIST_INTER_SAMPLE_AND_REF = './__clustered_score_dir_list_inter_sample_and_ref.pickle'
PATH_OF_DIR_LIST_INNER_REF = './__clustered_score_dir_list_inner_ref.pickle'
PATH_OF_DIR_LIST_INTER_REF = './__clustered_score_dir_list_inter_ref.pickle'


def _write_cluster_frame_core(dir_output, id_combination_list):
    # Create output folder.
    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)

    clustered_scores = []
    combination_idx = 0

    score_dtype = [
        ('index', 'u8'),
        ('keyword_a', 'O'),
        ('keyword_b', 'O'),
        ('index_a', 'u8'),
        ('index_b', 'u8'),
        ('global_accession_a', 'O'),
        ('global_accession_b', 'O'),
        ('cluster_id_a', 'u8'),
        ('cluster_id_b', 'u8'),
        ('cluster_name_a', 'O'),
        ('cluster_name_b', 'O'),
        ('score', 'f2'),
        ('matches', 'u2'),
        ('matched_peak_idx_a', 'O'),  # list
        ('matched_peak_idx_b', 'O')  # list
    ]

    for (keyword_a, cluster_id_a, cluster_name_a), (keyword_b, cluster_id_b, cluster_name_b) in id_combination_list:

        if cluster_id_a <= cluster_id_b:
            clustered_scores.append((
                combination_idx,  # 'index'
                keyword_a,  # 'keyword_a'
                keyword_b,  # 'keyword_b'
                0,  # 'index_a'
                0,  # 'index_b'
                '',  # 'global_accession_a'
                '',  # 'global_accession_b'
                cluster_id_a,  # 'cluster_id_a'
                cluster_id_b,  # 'cluster_id_b'
                cluster_name_a,  # 'cluster_name_a'
                cluster_name_b,  # 'cluster_name_b'
                0.0,  # 'score'
                0,  # 'matches'
                [],  # 'matched_peak_idx_a'
                []  # 'matched_peak_idx_b'
            ))
        else:
            # 'cluster_id_a' field should be smaller than 'cluster_id_b' so that swap them.
            LOGGER.warning('cluster_id_a > cluster_id_b')
            clustered_scores.append((
                combination_idx,  # 'index'
                keyword_b,  # 'keyword_a'
                keyword_a,  # 'keyword_b'
                0,  # 'index_a'
                0,  # 'index_b'
                '',  # 'global_accession_a'
                '',  # 'global_accession_b'
                cluster_id_b,  # 'cluster_id_a'
                cluster_id_a,  # 'cluster_id_b'
                cluster_name_b,  # 'cluster_name_a'
                cluster_name_a,  # 'cluster_name_b'
                0.0,  # 'score'
                0,  # 'matches'
                [],  # 'matched_peak_idx_a'
                []  # 'matched_peak_idx_b'
            ))

        if (combination_idx + 1) % 1000000 == 0:
            clustered_score_arr = np.array(clustered_scores, dtype=score_dtype)

            # Sort by cluster_id_a and cluster_id_b
            clustered_score_arr = clustered_score_arr[np.argsort(clustered_score_arr,
                                                                 order=['cluster_id_a', 'cluster_id_b'])]

            clustered_score_path = os.path.join(dir_output, f'{combination_idx - 999999}.npy')
            # Write cluster frame.
            with open(clustered_score_path, 'wb') as f:
                np.save(f, clustered_score_arr)
                f.flush()

            clustered_scores = []

        combination_idx += 1

    clustered_score_arr = np.array(clustered_scores, dtype=score_dtype)
    
    # Sort by cluster_id_a and cluster_id_b
    clustered_score_arr = clustered_score_arr[np.argsort(clustered_score_arr, order=['cluster_id_a', 'cluster_id_b'])]

    clustered_score_path = os.path.join(dir_output, f'{int(combination_idx / 1000000) * 1000000}.npy')
    # Write cluster frame.
    with open(clustered_score_path, 'wb') as f:
        np.save(f, clustered_score_arr)
        f.flush()


def create_cluster_frame_for_grouped_spectra(sample_metadata_dir,
                                             ref_metadata_dir,
                                             output_parent_dir,
                                             calculate_inner_sample=True,
                                             calculate_inter_sample=True,
                                             calculate_inter_sample_and_ref=True,
                                             calculate_inner_ref=True,
                                             calculate_inter_ref=True,
                                             export_metadata_tsv=True):
    sample_metadata_path_vs_idx_list = get_npy_metadata_paths(sample_metadata_dir)
    ref_metadata_path_vs_idx_list = get_npy_metadata_paths(ref_metadata_dir)

    all_cluster_name_list = []

    # id_information: [(keyword_a, cluster_id_a, cluster_name_a),
    #                  (keyword_b, cluster_id_b, cluster_name_b),
    #                  (keyword_c, cluster_id_c, cluster_name_c), ...]
    id_information_dict_by_dataset = {}
    cluster_id_start = 0

    # Define cluster name and cluster id of sample. -----------------------------------------------
    for sample_metadata_path, _ in sample_metadata_path_vs_idx_list:
        filename = os.path.splitext(os.path.basename(sample_metadata_path))[0]
        metadata_arr = np.load(sample_metadata_path, allow_pickle=True)

        # Add cluster name: dataset filename + | + index
        metadata_arr['cluster_name'] = np.char.add(f'{filename}|', metadata_arr['index'].astype(str))
        # Add cluster id (integer index)
        metadata_arr['cluster_id'] = np.array(list(range(cluster_id_start, cluster_id_start + len(metadata_arr))))

        # Add id_information to id_information_dict_by_dataset
        id_information = [(_keyword, _cluster_id, _cluster_name) for _keyword, _cluster_id, _cluster_name
                          in zip(metadata_arr['keyword'], metadata_arr['cluster_id'], metadata_arr['cluster_name'])]
        id_information_dict_by_dataset[filename] = id_information

        all_cluster_name_list.extend(list(metadata_arr['cluster_name']))
        cluster_id_start += len(metadata_arr)

        # Update metadata
        with open(sample_metadata_path, 'wb') as f:
            np.save(f, metadata_arr)
            f.flush()
        
        if export_metadata_tsv:
            tsv_path = os.path.splitext(sample_metadata_path)[0] + '.tsv'
            df = pd.DataFrame.from_records(metadata_arr)
            df.to_csv(tsv_path, sep='\t', index=True)
    # ---------------------------------------------------------------------------------------------

    # Define cluster name and cluster id of reference. --------------------------------------------
    for ref_metadata_path, _ in ref_metadata_path_vs_idx_list:
        filename = os.path.splitext(os.path.basename(ref_metadata_path))[0]
        metadata_arr = np.load(ref_metadata_path, allow_pickle=True)

        cluster_name_body_arr = np.where(metadata_arr['inchikey'] == '',
                                         metadata_arr['index'].astype(str),
                                         metadata_arr['inchikey'].astype(str))

        # Add cluster name: dataset filename + | + index or inchikey
        metadata_arr['cluster_name'] = np.char.add(f'{filename}|', cluster_name_body_arr)

        # Add cluster id
        cluster_name_body_df = pd.DataFrame(metadata_arr['cluster_name'], columns=['cluster_name'])
        cluster_name_body_df_unique = pd.DataFrame(cluster_name_body_df['cluster_name'].unique(),
                                                   columns=['cluster_name'])
        cluster_name_body_df_unique['cluster_id'] = list(range(cluster_id_start, cluster_id_start + len(cluster_name_body_df_unique)))
        cluster_name_body_df_unique['keyword'] = metadata_arr[0]['keyword']
        cluster_name_body_df = pd.merge(cluster_name_body_df, cluster_name_body_df_unique, on='cluster_name')
        metadata_arr['cluster_id'] = cluster_name_body_df['cluster_id'].values

        # Add id_information to id_information_dict_by_dataset
        id_information = [(_keyword, _cluster_id, _cluster_name) for _keyword, _cluster_id, _cluster_name
                          in zip(cluster_name_body_df_unique['keyword'],
                                 cluster_name_body_df_unique['cluster_id'],
                                 cluster_name_body_df_unique['cluster_name'])]
        id_information_dict_by_dataset[filename] = id_information

        all_cluster_name_list.extend(list(cluster_name_body_df_unique['cluster_name']))
        cluster_id_start += len(cluster_name_body_df_unique)

        # Update metadata
        with open(ref_metadata_path, 'wb') as f:
            np.save(f, metadata_arr)
            f.flush()

        if export_metadata_tsv:
            tsv_path = os.path.splitext(ref_metadata_path)[0] + '.tsv'
            df = pd.DataFrame.from_records(metadata_arr)
            df.to_csv(tsv_path, sep='\t', index=True)
    # ---------------------------------------------------------------------------------------------

    # Write all_cluster_name_list to a pickle file.
    with open(CLUSTER_NAME_LIST_PATH, 'wb') as f:
        pickle.dump(all_cluster_name_list, f)

    dataset_combinations = []
    # Add inner sample combination: sample dataset x vs sample dataset x scores. ----------
    if calculate_inner_sample:
        output_dir_list = []
        for path, _ in sample_metadata_path_vs_idx_list:
            filename_a = os.path.splitext(os.path.basename(path))[0]
            output_folder_name = f'{filename_a}_vs_{filename_a}'
            output_dir = os.path.join(output_parent_dir, output_folder_name)
            dataset_combinations.append((filename_a, filename_a, output_dir))
            output_dir_list.append(output_dir)

        # Save output_dir_list
        with open(PATH_OF_DIR_LIST_INNER_SAMPLE, 'wb') as f:
            pickle.dump(output_dir_list, f)
    else:
        with open(PATH_OF_DIR_LIST_INNER_SAMPLE, 'wb') as f:
            pickle.dump([], f)
    # -------------------------------------------------------------------------------------

    # Add inter sample combination: sample dataset x vs sample dataset y scores. ----------
    if calculate_inter_sample:
        output_dir_list = []
        for (path_a, _), (path_b, _) in list(itertools.combinations(sample_metadata_path_vs_idx_list, 2)):
            filename_a = os.path.splitext(os.path.basename(path_a))[0]
            filename_b = os.path.splitext(os.path.basename(path_b))[0]
            output_folder_name = f'{filename_a}_vs_{filename_b}'
            output_dir = os.path.join(output_parent_dir, output_folder_name)
            dataset_combinations.append((filename_a, filename_b, output_dir))
            output_dir_list.append(output_dir)

        # Save output_dir_list
        with open(PATH_OF_DIR_LIST_INTER_SAMPLE, 'wb') as f:
            pickle.dump(output_dir_list, f)
    else:
        with open(PATH_OF_DIR_LIST_INTER_SAMPLE, 'wb') as f:
            pickle.dump([], f)
    # -------------------------------------------------------------------------------------

    # Add inter sample and reference combination: sample dataset x vs reference dataset x scores. -------
    if calculate_inter_sample_and_ref:
        output_dir_list = []
        for (path_a, _), (path_b, _) in list(itertools.product(sample_metadata_path_vs_idx_list,
                                                               ref_metadata_path_vs_idx_list)):
            filename_a = os.path.splitext(os.path.basename(path_a))[0]
            filename_b = os.path.splitext(os.path.basename(path_b))[0]
            output_folder_name = f'{filename_a}_vs_{filename_b}'
            output_dir = os.path.join(output_parent_dir, output_folder_name)
            dataset_combinations.append((filename_a, filename_b, output_dir))
            output_dir_list.append(output_dir)

        # Save output_dir_list
        with open(PATH_OF_DIR_LIST_INTER_SAMPLE_AND_REF, 'wb') as f:
            pickle.dump(output_dir_list, f)
    else:
        with open(PATH_OF_DIR_LIST_INTER_SAMPLE_AND_REF, 'wb') as f:
            pickle.dump([], f)
    # ---------------------------------------------------------------------------------------------------

    # Add inner reference combination: reference dataset x vs reference dataset x scores. ---------------
    if calculate_inner_ref:
        output_dir_list = []
        for path, _ in ref_metadata_path_vs_idx_list:
            filename_a = os.path.splitext(os.path.basename(path))[0]
            output_folder_name = f'{filename_a}_vs_{filename_a}'
            output_dir = os.path.join(output_parent_dir, output_folder_name)
            dataset_combinations.append((filename_a, filename_a, output_dir))
            output_dir_list.append(output_dir)

        # Save output_dir_list
        with open(PATH_OF_DIR_LIST_INNER_REF, 'wb') as f:
            pickle.dump(output_dir_list, f)
    else:
        with open(PATH_OF_DIR_LIST_INNER_REF, 'wb') as f:
            pickle.dump([], f)
    # ---------------------------------------------------------------------------------------------------

    # Add inter reference combination: reference dataset x vs reference dataset y scores. ---------------
    if calculate_inter_ref:
        output_dir_list = []
        for (path_a, _), (path_b, _) in list(itertools.combinations(ref_metadata_path_vs_idx_list, 2)):
            filename_a = os.path.splitext(os.path.basename(path_a))[0]
            filename_b = os.path.splitext(os.path.basename(path_b))[0]
            output_folder_name = f'{filename_a}_vs_{filename_b}'
            output_dir = os.path.join(output_parent_dir, output_folder_name)
            dataset_combinations.append((filename_a, filename_b, output_dir))
            output_dir_list.append(output_dir)

        # Save output_dir_list
        with open(PATH_OF_DIR_LIST_INTER_REF, 'wb') as f:
            pickle.dump(output_dir_list, f)
    else:
        with open(PATH_OF_DIR_LIST_INTER_REF, 'wb') as f:
            pickle.dump([], f)
    # ---------------------------------------------------------------------------------------------------

    # Create cluster frame -------------------------------------------------------------------
    for filename_a, filename_b, output_dir in dataset_combinations:
        # Make a output folder
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        if filename_a == filename_b:
            id_information = id_information_dict_by_dataset[filename_a]
            id_combination_list = list(itertools.combinations(id_information, 2))
        else:
            id_information_a = id_information_dict_by_dataset[filename_a]
            id_information_b = id_information_dict_by_dataset[filename_b]
            id_combination_list = list(itertools.product(id_information_a, id_information_b))

        _write_cluster_frame_core(output_dir, id_combination_list)
    # ----------------------------------------------------------------------------------------
