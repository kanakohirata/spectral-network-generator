from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import os
from utils import get_paths


LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


def cluster_grouped_score_based_on_cluster_id_core(score_path, clustered_score_paths):
    LOGGER.info(f'Cluster {os.path.basename(score_path)}')
    # Load score array
    score_arr = np.load(score_path, allow_pickle=True)

    # Filter out scores with cluster_id_a == cluster_id_b ---------------------
    mask_to_filter = score_arr['cluster_id_a'] != score_arr['cluster_id_b']
    if not np.any(mask_to_filter):
        return
    
    score_arr = score_arr[mask_to_filter]
    # ------------------------------------------------------------------------

    # Rearrange score_arr (not clustered) in order that score_arr['cluster_id_a'] < score_arr['cluster_id_b']
    mask_to_replace = score_arr['cluster_id_a'] > score_arr['cluster_id_b']
    if np.any(mask_to_replace):
        LOGGER.info("Rearrange score_arr in order that score_arr['cluster_id_a'] < score_arr['cluster_id_b']")

        _arr_to_replace = np.zeros(score_arr[mask_to_replace].shape[0], dtype=score_arr.dtype)

        # Swap keyword_a and keyword_b
        _arr_to_replace['keyword_a'] = score_arr[mask_to_replace]['keyword_b']
        _arr_to_replace['keyword_b'] = score_arr[mask_to_replace]['keyword_a']
        # Swap index_a and index_b
        _arr_to_replace['index_a'] = score_arr[mask_to_replace]['index_b']
        _arr_to_replace['index_b'] = score_arr[mask_to_replace]['index_a']
        # Swap cluster_id_a and cluster_id_b
        _arr_to_replace['cluster_id_a'] = score_arr[mask_to_replace]['cluster_id_b']
        _arr_to_replace['cluster_id_b'] = score_arr[mask_to_replace]['cluster_id_a']
        # Swap cluster_name_a and cluster_name_b
        _arr_to_replace['cluster_name_a'] = score_arr[mask_to_replace]['cluster_name_b']
        _arr_to_replace['cluster_name_b'] = score_arr[mask_to_replace]['cluster_name_a']
        # Swap global_accession_a and global_accession_b
        _arr_to_replace['global_accession_a'] = score_arr[mask_to_replace]['global_accession_b']
        _arr_to_replace['global_accession_b'] = score_arr[mask_to_replace]['global_accession_a']
        # Swap matched_peak_idx_a and matched_peak_idx_b
        _arr_to_replace['matched_peak_idx_a'] = score_arr[mask_to_replace]['matched_peak_idx_b']
        _arr_to_replace['matched_peak_idx_b'] = score_arr[mask_to_replace]['matched_peak_idx_a']

        _arr_to_replace['score'] = score_arr[mask_to_replace]['score']
        _arr_to_replace['matches'] = score_arr[mask_to_replace]['matches']
        _arr_to_replace['index'] = score_arr[mask_to_replace]['index']

        score_arr[mask_to_replace] = _arr_to_replace
        del _arr_to_replace

    # Sort by similarity score and number of matched peaks
    score_arr = score_arr[np.argsort(score_arr, order=['score', 'matches'])[::-1]]
    # Extract scores with unique combination of cluster_id_a and cluster_id_b
    temp_arr = np.array(score_arr[['cluster_id_a', 'cluster_id_b']].tolist())
    unique_indices = np.unique(temp_arr, axis=0, return_index=True)[1]
    score_arr = score_arr[unique_indices]

    # Sort by cluster_id_a and cluster_id_b
    score_arr = score_arr[np.argsort(score_arr, order=['cluster_id_a', 'cluster_id_b'])]
    
    for clustered_score_path in clustered_score_paths:
        # Load clustered score array
        clustered_score_arr = np.load(clustered_score_path, allow_pickle=True)

        if clustered_score_arr is None:
            continue
        # Extract scores where score_arr[['cluster_id_a', 'cluster_id_b']] is in clustered_score_arr[['cluster_id_a', 'cluster_id_b']]
        mask_is_in_clustered = np.isin(score_arr[['cluster_id_a', 'cluster_id_b']],
                                    clustered_score_arr[['cluster_id_a', 'cluster_id_b']])
        if not np.any(mask_is_in_clustered):
            continue

        score_arr_temp = score_arr[mask_is_in_clustered]
        _x = score_arr[~mask_is_in_clustered]
        # ------------------------------------------------------------------------------------------------------------------

        # Extract scores where clustered_score_arr[['cluster_id_a', 'cluster_id_b']] is in score_arr[['cluster_id_a', 'cluster_id_b']]
        mask_is_in_not_clustered = np.isin(clustered_score_arr[['cluster_id_a', 'cluster_id_b']],
                                        score_arr_temp[['cluster_id_a', 'cluster_id_b']])
        if not np.any(mask_is_in_not_clustered):
            continue

        clustered_score_arr_temp = clustered_score_arr[mask_is_in_not_clustered]
        # ------------------------------------------------------------------------------------------------------------------

        _test = clustered_score_arr_temp[['cluster_id_a', 'cluster_id_b']]\
                    == score_arr_temp[['cluster_id_a', 'cluster_id_b']]
        if not np.all(_test):
            raise ValueError('Order of "cluster_id_a" and "cluster_id_b" must be the same between arrays.')

        score_arr_temp['index'] = clustered_score_arr_temp['index']

        # Match the order of fields in clustered_score_arr_temp and score_arr_temp.
        fields = [x for x in clustered_score_arr_temp.dtype.names]
        score_arr_temp = score_arr_temp[fields]

        # Get bool array indicating whether clustered_score_arr_temp['score'] < score_arr_temp['score']
        mask_score = (clustered_score_arr_temp['score'] < score_arr_temp['score']) \
                    | ((clustered_score_arr_temp['score'] == score_arr_temp['score'])
                        & (clustered_score_arr_temp['matches'] <= score_arr_temp['matches']))

        if not np.any(mask_score):
            continue

        # Replace clustered_score_arr_temp where clustered_score_arr_temp['score'] < score_arr_temp['score']
        # with score_arr_temp
        clustered_score_arr_temp[mask_score] = score_arr_temp[mask_score]

        # Update clustered_score_arr
        clustered_score_arr[mask_is_in_not_clustered] = clustered_score_arr_temp
        clustered_score_folder_name = os.path.basename(os.path.dirname(clustered_score_path))
        LOGGER.info(f'Update {clustered_score_folder_name}/{os.path.basename(clustered_score_path)}')
        with open(clustered_score_path, 'wb') as f:
            np.save(f, clustered_score_arr)
            f.flush()


def cluster_grouped_score_based_on_cluster_id(parent_score_dir, parent_clustered_score_dir):
    # Get paths of not clustered score.
    score_dir_name_vs_path_list = get_paths.get_folder_name_vs_path_list(parent_score_dir)
    clustered_score_dir_name_vs_path_dict = get_paths.get_folder_name_vs_path_dict(parent_clustered_score_dir)

    for score_dir_name, score_dir_path in score_dir_name_vs_path_list:
        if score_dir_name not in clustered_score_dir_name_vs_path_dict:
            continue

        # Get clustered score paths.
        clustered_score_dir_path = clustered_score_dir_name_vs_path_dict[score_dir_name]
        clustered_score_paths = get_paths.get_npy_paths(clustered_score_dir_path)

        score_paths = get_paths.get_npy_paths(score_dir_path)
        for score_path in score_paths:
            cluster_grouped_score_based_on_cluster_id_core(score_path, clustered_score_paths)
