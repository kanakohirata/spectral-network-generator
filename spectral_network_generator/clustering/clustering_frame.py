import h5py
import itertools
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import numpy.lib.recfunctions as rfn
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

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


CLUSTER_NAME_LIST_PATH = './__cluster_name_list.pickle'
PATH_OF_DIR_LIST_INNER_SAMPLE = './__clustered_score_dir_list_inner_sample.pickle'
PATH_OF_DIR_LIST_INTER_SAMPLE = './__clustered_score_dir_list_inter_sample.pickle'
PATH_OF_DIR_LIST_INTER_SAMPLE_AND_REF = './__clustered_score_dir_list_inter_sample_and_ref.pickle'
PATH_OF_DIR_LIST_INNER_REF = './__clustered_score_dir_list_inner_ref.pickle'
PATH_OF_DIR_LIST_INTER_REF = './__clustered_score_dir_list_inter_ref.pickle'


def add_cluster_id(score_path='./score.h5'):
    with h5py.File(score_path, 'a') as h5:
        dset = h5['clustered_score']
        mask_a = np.where(dset.fields('inchikey_a')[()] != b'')
        mask_b = np.where(dset.fields('inchikey_b')[()] != b'')
        inchikey_arr_a = dset.fields('inchikey_a')
        inchikey_arr_b = dset.fields('inchikey_b')
        df = pd.DataFrame(dset[()][['global_accession_a', 'global_accession_b', 'inchikey_a', 'inchikey_b']],
                          columns=['global_accession_a', 'global_accession_b', 'inchikey_a', 'inchikey_b'])
        inchikey_arr = np.unique(np.hstack((inchikey_arr_a[mask_a], inchikey_arr_b[mask_b]))).astype('S')
        index_arr = np.arange(inchikey_arr.shape[0]).astype('S')
        cluster_id_arr = np.core.defchararray.add(np.core.defchararray.add(index_arr, b'|'), inchikey_arr)
        df_cluster_id = pd.DataFrame({'inchikey_a': inchikey_arr, 'cluster_id_a': cluster_id_arr})
        df = pd.merge(df, df_cluster_id, on='inchikey_a', how='left')
        df_cluster_id.rename(columns={'inchikey_a': 'inchikey_b', 'cluster_id_a': 'cluster_id_b'}, inplace=True)
        df = pd.merge(df, df_cluster_id, on='inchikey_b', how='left')

        del df_cluster_id

        df['cluster_id_a'].fillna(df['global_accession_a'], inplace=True)
        df['cluster_id_b'].fillna(df['global_accession_b'], inplace=True)

        cluster_id_arr_a = df['cluster_id_a'].values.astype('S')
        cluster_id_arr_b = df['cluster_id_b'].values.astype('S')

        del df

        score_arr = rfn.append_fields(dset[()], 'cluster_id_a', cluster_id_arr_a, usemask=False)
        score_arr = rfn.append_fields(score_arr, 'cluster_id_b', cluster_id_arr_b, usemask=False)

        del h5['clustered_score']
        h5.create_dataset('clustered_score', data=score_arr, shape=score_arr.shape, maxshape=(None,))


def create_cluster_frame():
    with h5py.File('./spectrum_metadata.h5', 'r') as h5:
        dset = h5['filtered/metadata']
        mask_sample = dset.fields('tag')[()] == b'sample'
        mask_ref = dset.fields('tag')[()] == b'ref'
        sample_arr = dset.fields('global_accession')[()][mask_sample]
        ref_arr = dset[()][['global_accession', 'inchikey']][mask_ref]
        
        cluster_ids = []
        if sample_arr.size:
            cluster_ids += sample_arr.ravel().tolist()
        if ref_arr.size:
            ref_arr = np.where(ref_arr['inchikey'] == b'', ref_arr['global_accession'], ref_arr['inchikey'])
            ref_arr = np.unique(ref_arr)

            cluster_ids += ref_arr.tolist()
    
    with open('./cluster_ids.pickle', 'wb') as f:
        pickle.dump(cluster_ids, f)

    if mask_sample.size == len(cluster_ids):
        # SpecNetGenConfig.is_clustering_required = False
        return False

    max_length_of_cluster_id = max(len(x) for x in cluster_ids)
    cluster_index_vs_id_list = [(idx, id_) for idx, id_ in enumerate(cluster_ids)]
    clustered_scores = []
    combination_idx = 0
    for (cluster_index_a, cluster_id_a), (cluster_index_b, cluster_id_b) in itertools.combinations(cluster_index_vs_id_list, 2):
        # append tuple(combination_idx, cluster_idx_a, cluster_idx_b, score, number of matched peaks, cluster_id_a, cluster_id_b)
        clustered_scores.append((combination_idx, cluster_index_a, cluster_index_b, 0, 0, 0.0, 0, cluster_id_a, cluster_id_b))
        if (combination_idx + 1) % 1000000 == 0:
            clustered_score_arr = np.array(clustered_scores,
                                           dtype=[('index', 'u8'),
                                                  ('cluster_index_a', 'u8'), ('cluster_index_b', 'u8'),
                                                  ('index_a', 'u8'), ('index_b', 'u8'),
                                                  ('score', 'f2'), ('matches', 'u2'),
                                                  ('cluster_id_a', f'S{max_length_of_cluster_id}'),
                                                  ('cluster_id_b', f'S{max_length_of_cluster_id}')])
            
            clustered_score_path = f'./scores/clustered_scores/{combination_idx - 999999}.npy'
            with open(clustered_score_path, 'wb') as f:
                np.save(f, clustered_score_arr)
            clustered_scores = []

        combination_idx += 1

    clustered_score_arr = np.array(clustered_scores,
                                   dtype=[('index', 'u8'),
                                          ('cluster_index_a', 'u8'), ('cluster_index_b', 'u8'),
                                          ('index_a', 'u8'), ('index_b', 'u8'),
                                          ('score', 'f2'), ('matches', 'u2'),
                                          ('cluster_id_a', f'S{max_length_of_cluster_id}'),
                                          ('cluster_id_b', f'S{max_length_of_cluster_id}')])
    clustered_score_path = f'./scores/clustered_scores/{int(combination_idx / 1000000) * 1000000}.npy'
    with open(clustered_score_path, 'wb') as f:
        np.save(f, clustered_score_arr)

    # SpecNetGenConfig.is_clustering_required = False
    return True


def _write_cluster_frame_core_old(dir_output, cluster_index_and_cluster_id_combination_list, max_length_of_cluster_id):
    # Create output folder.
    if not os.path.isdir(dir_output):
        os.makedirs(dir_output)

    clustered_scores = []
    combination_idx = 0

    for (cluster_index_a, cluster_id_a), (cluster_index_b, cluster_id_b) in\
            cluster_index_and_cluster_id_combination_list:

        clustered_scores.append((
            combination_idx,  # 'index'
            cluster_index_a,  # 'cluster_index_a'
            cluster_index_b,  # 'cluster_index_b'
            0,  # 'index_a'
            0,  # 'index_b'
            0.0,  # 'score'
            0,  # 'matches'
            cluster_id_a,  # 'cluster_id_a'
            cluster_id_b  # 'cluster_id_b'
        ))

        if (combination_idx + 1) % 1000000 == 0:
            clustered_score_arr = np.array(clustered_scores,
                                           dtype=[('index', 'u8'),
                                                  ('cluster_index_a', 'u8'), ('cluster_index_b', 'u8'),
                                                  ('index_a', 'u8'), ('index_b', 'u8'),
                                                  ('score', 'f2'), ('matches', 'u2'),
                                                  ('cluster_id_a', f'S{max_length_of_cluster_id}'),
                                                  ('cluster_id_b', f'S{max_length_of_cluster_id}')])

            clustered_score_path = os.path.join(dir_output, f'{combination_idx - 999999}.npy')
            with open(clustered_score_path, 'wb') as f:
                np.save(f, clustered_score_arr)
            clustered_scores = []

        combination_idx += 1

    clustered_score_arr = np.array(clustered_scores,
                                   dtype=[('index', 'u8'),
                                          ('cluster_index_a', 'u8'), ('cluster_index_b', 'u8'),
                                          ('index_a', 'u8'), ('index_b', 'u8'),
                                          ('score', 'f2'), ('matches', 'u2'),
                                          ('cluster_id_a', f'S{max_length_of_cluster_id}'),
                                          ('cluster_id_b', f'S{max_length_of_cluster_id}')])

    clustered_score_path = os.path.join(dir_output, f'{int(combination_idx / 1000000) * 1000000}.npy')
    with open(clustered_score_path, 'wb') as f:
        np.save(f, clustered_score_arr)


def create_cluster_frame_for_grouped_spectra_old():


    with h5py.File('./spectrum_metadata.h5', 'r') as h5:
        all_cluster_id_list = []

        # Create a list of tuples of sample cluster index and sample cluster id.
        # Cluster index is just an integer index.
        # Cluster id is dataset keyword plus global_accession: 'sample|0|sample_aaa.msp'
        sample_cluster_index_and_cluster_id_list = []
        count = 0
        sample_dset = h5['grouped/sample']
        sample_accession_arr = sample_dset['global_accession'][()]
        sample_accession_string_length = max(len(x) for x in sample_accession_arr) + len(b'sample|')
        sample_accession_arr = sample_accession_arr.astype(f'S{sample_accession_string_length}')
        sample_cluster_id_arr = np.core.defchararray.add(b'sample|', sample_accession_arr)
        for sample_cluster_id in sample_cluster_id_arr:
            sample_cluster_index_and_cluster_id_list.append((count, sample_cluster_id))
            all_cluster_id_list.append(sample_cluster_id)
            count += 1

        # Create a dictionary which has dataset keyword as key and
        # a list of tuples containing reference cluster index and reference cluster id as value.
        # Cluster id is dataset keyword plus inchikey or global_accession: 'Benzenoids|VYFYYTLLBUKUHU-UHFFFAOYSA-N'
        ref_keyword_vs_cluster_index_and_cluster_id_dict = {}
        for dataset_keyword in h5['grouped'].keys():
            if dataset_keyword == 'sample':
                continue
            ref_cluster_index_and_cluster_id_list = []
            dset = h5[f'grouped/{dataset_keyword}']
            ref_accession_arr = np.where(dset['inchikey'][()] == b'',
                                         dset['global_accession'][()],
                                         dset['inchikey'][()])
            ref_accession_arr = np.unique(ref_accession_arr)
            dataset_keyword_byte = f'{dataset_keyword}|'.encode('utf8')
            ref_accession_string_length = max(len(x) for x in ref_accession_arr) + len(dataset_keyword_byte)
            ref_accession_arr = ref_accession_arr.astype(f'S{ref_accession_string_length}')
            ref_cluster_id_arr = np.core.defchararray.add(dataset_keyword_byte, ref_accession_arr)
            for ref_cluster_id in ref_cluster_id_arr:
                ref_cluster_index_and_cluster_id_list.append((count, ref_cluster_id))
                sample_cluster_id.append(ref_cluster_id)
                count += 1

            ref_keyword_vs_cluster_index_and_cluster_id_dict[dataset_keyword] = ref_cluster_index_and_cluster_id_list

    # Export all_cluster_id_list
    with open('./cluster_ids.pickle', 'wb') as f:
        pickle.dump(all_cluster_id_list, f)

    # Get maximum length of cluster id.
    max_length_of_cluster_id = max(len(x) for x in ref_cluster_id)

    # Create clustered score frame for samples: sample vs sample or sample vs reference ----------------------------
    sample_cluster_index_and_cluster_id_combination_list =\
        list(itertools.combinations(sample_cluster_index_and_cluster_id_list, 2))

    for dataset_keyword, ref_cluster_index_and_cluster_id_list in\
            ref_keyword_vs_cluster_index_and_cluster_id_dict.items():
        sample_cluster_index_and_cluster_id_combination_list +=\
            list(itertools.product(sample_cluster_index_and_cluster_id_list, ref_cluster_index_and_cluster_id_list))

    _write_cluster_frame_core('./scores/grouped_and_clustered_scores/sample',
                              sample_cluster_index_and_cluster_id_combination_list,
                              max_length_of_cluster_id)
    # --------------------------------------------------------------------------------------------------------------

    # Create clustered score frame of references for each dataset: reference vs reference --------------------------
    for dataset_keyword, ref_cluster_index_and_cluster_id_list in\
            ref_keyword_vs_cluster_index_and_cluster_id_dict.items():

        output_dir = f'./scores/grouped_and_clustered_scores/{dataset_keyword}'

        ref_cluster_index_and_cluster_id_combination_list =\
            itertools.combinations(ref_cluster_index_and_cluster_id_list, 2)

        _write_cluster_frame_core(output_dir,
                                  ref_cluster_index_and_cluster_id_combination_list,
                                  max_length_of_cluster_id)
    # --------------------------------------------------------------------------------------------------------------

    return max_length_of_cluster_id


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

    # id_information: [(cluster_id_a, cluster_name_a),
    #                  (cluster_id_b, cluster_name_b),
    #                  (cluster_id_c, cluster_name_c), ...]
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


if __name__ == '__main__':
    create_cluster_frame()
