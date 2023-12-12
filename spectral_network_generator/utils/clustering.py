import h5py
import itertools
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import numpy.lib.recfunctions as rfn
import os
import pandas as pd
import pickle

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


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


def _write_cluster_frame_core(dir_output, cluster_index_and_cluster_id_combination_list, max_length_of_cluster_id):
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


def create_cluster_frame_for_grouped_spectra():
    with h5py.File('./spectrum_metadata.h5', 'r') as h5:
        all_cluster_index_and_cluster_id_list = []

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
            all_cluster_index_and_cluster_id_list.append((count, sample_cluster_id))
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
                all_cluster_index_and_cluster_id_list.append((count, ref_cluster_id))
                count += 1

            ref_keyword_vs_cluster_index_and_cluster_id_dict[dataset_keyword] = ref_cluster_index_and_cluster_id_list

    # Get maximum length of cluster id.
    max_length_of_cluster_id = max(len(x[1]) for x in all_cluster_index_and_cluster_id_list)

    # Create clustered score frame for samples: sample vs sample or sample vs reference ----------------------------
    sample_cluster_index_and_cluster_id_combination_list =\
        list(itertools.combinations(sample_cluster_index_and_cluster_id_list, 2))

    for dataset_keyword, ref_cluster_index_and_cluster_id_list in\
            ref_keyword_vs_cluster_index_and_cluster_id_dict.items():
        sample_cluster_index_and_cluster_id_combination_list +=\
            list(itertools.product(sample_cluster_index_and_cluster_id_list, ref_cluster_index_and_cluster_id_list))

    _write_cluster_frame_core('./scores/grouped_scores/sample',
                              sample_cluster_index_and_cluster_id_combination_list,
                              max_length_of_cluster_id)
    # --------------------------------------------------------------------------------------------------------------

    # Create clustered score frame of references for each dataset: reference vs reference --------------------------
    for dataset_keyword, ref_cluster_index_and_cluster_id_list in\
            ref_keyword_vs_cluster_index_and_cluster_id_dict.items():

        output_dir = f'./scores/grouped_scores/{dataset_keyword}'

        ref_cluster_index_and_cluster_id_combination_list =\
            itertools.combinations(ref_cluster_index_and_cluster_id_list, 2)

        _write_cluster_frame_core(output_dir,
                                  ref_cluster_index_and_cluster_id_combination_list,
                                  max_length_of_cluster_id)
    # --------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    create_cluster_frame()
