import h5py
import numpy as np
import os
import re
import shutil
from utils import get_paths


H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


def initialize_score_hdf5(path='./score.h5'):
    with h5py.File(path, 'w') as h5:
        pass


def initialize_score_files(dir_path='./scores',
                           clustered_dir_name='clustered',
                           grouped_dir_name='grouped',
                           raw_dir_name='raw'):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    else:
        for f in os.listdir(dir_path):
            p = os.path.join(dir_path, f)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.splitext(f)[1] == '.npy':
                os.remove(p)

    dir_path_raw = os.path.join(dir_path, raw_dir_name)
    if not os.path.isdir(dir_path_raw):
        os.makedirs(dir_path_raw)

    dir_path_clustered = os.path.join(dir_path, clustered_dir_name)
    if not os.path.isdir(dir_path_clustered):
        os.makedirs(dir_path_clustered)

    dir_path_grouped = os.path.join(dir_path, grouped_dir_name)
    if not os.path.isdir(dir_path_grouped):
        os.makedirs(dir_path_grouped)


def get_chunks(key, db_chunk_size=10000, path='./score.h5'):
    """
    Parameters
    ----------
    key: str
        path to dataset
    db_chunk_size: int or None
        Number of compounds loaded at a time
    path: str

    Returns
    -------
    Iterator[numpy.array]
    """

    with h5py.File(path, 'r') as h5:
        dset = h5[key]

        if db_chunk_size:
            chunk_num = int(dset.size / db_chunk_size) + 1

        else:
            db_chunk_size = dset.size
            chunk_num = 1

        for i in range(chunk_num):
            start = db_chunk_size * i
            end = db_chunk_size * (i + 1)

            yield dset[start:end], start, end


def iter_score_array(dir_path='./scores', return_path=True, return_index=True):
    score_paths = []
    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and re.match(r'\d+-\d+_vs_\d+-\d+\.npy', filename):
            
            indexes_a, indexes_b = filename.split('_vs_')
            idx_a = int(indexes_a.split('-')[0])
            idx_b = int(indexes_b.split('-')[0])

            score_paths.append((path, idx_a, idx_b))

    if not score_paths:
        if return_path and return_index:
            return None, None, None, None
        elif return_path:
            return None, None
        elif return_index:
            return None, None, None
        else:
            return None
    
    score_paths.sort(key=lambda x: (x[1], x[2]))

    for path, idx_a, idx_b in score_paths:
        arr = np.load(path)

        if return_path and return_index:
            yield arr, path, idx_a, idx_b
        elif return_path:
            yield arr, path
        elif return_index:
            yield arr, idx_a, idx_b
        else:
            yield arr


def iter_clustered_score_array(dir_path='./scores/clustered_scores', return_path=True, return_index=True):
    score_paths = []
    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and re.match(r'\d+\.npy', filename):  
            idx = int(os.path.splitext(filename)[0])
            score_paths.append((path, idx))

    if not score_paths:
        if return_path and return_index:
            return None, None, None
        elif return_path or return_index:
            return None, None
        else:
            return None

    score_paths.sort(key=lambda x: x[1])

    for path, idx in score_paths:
        arr = np.load(path)

        if return_path and return_index:
            yield arr, path, idx
        elif return_path:
            yield arr, path
        elif return_index:
            yield arr, idx
        else:
            yield arr


def iter_grouped_and_clustered_score_array(parent_dir_path='./scores/grouped_and_clustered_scores', grouped_metadata_key='grouped', return_path=True, return_index=True):
    # Get sample score folder
    dir_paths = [os.path.join(parent_dir_path, 'sample')]

    # Get reference score folders
    with h5py.File('./spectrum_matadata.h5', 'r') as h5:
        for k in h5[grouped_metadata_key].key():
            if k != 'sample':
                dir_paths.append(os.path.join(parent_dir_path, k))

    for dir_path in dir_paths:
        score_paths = []
        for filename in os.listdir(dir_path):
            path = os.path.join(dir_path, filename)
            if os.path.isfile(path) and re.match(r'\d+\.npy', filename):  
                idx = int(os.path.splitext(filename)[0])
                score_paths.append((path, idx))

        if not score_paths:
            if return_path and return_index:
                return None, None, None
            elif return_path or return_index:
                return None, None
            else:
                return None

        score_paths.sort(key=lambda x: x[1])

        for path, idx in score_paths:
            arr = np.load(path)

            if return_path and return_index:
                yield arr, path, idx
            elif return_path:
                yield arr, path
            elif return_index:
                yield arr, idx
            else:
                yield arr


def adjust_string_length(data, max_lengths):
    adjusted_data = np.copy(data)

    dtype = [('index', 'u8'),
             ('cluster_index_a', 'u8'), ('cluster_index_b', 'u8'),
             ('index_a', 'u8'), ('index_b', 'u8'),
             ('score', 'f2'), ('matches', 'u2'),
             ('cluster_id_a', f'S{max_lengths["cluster_id_a"]}'),
             ('cluster_id_b', f'S{max_lengths["cluster_id_b"]}')]
    
    adjusted_data = adjusted_data.astype(dtype)

    return adjusted_data


def resize_clustered_score_file(row_size, dir_path='./scores/clustered_scores'):
    # Make "temp" folder and move score files to the folder.
    temp_folder = os.path.join(dir_path, 'temp')
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    score_filenames = [f for f in os.listdir(dir_path) if re.match(r'\d+\.npy', f)]
    score_paths = []
    for filename in score_filenames:
        shutil.move(os.path.join(dir_path, filename), temp_folder)

        # Get new path
        path = os.path.join(temp_folder, filename)
        idx = int(os.path.splitext(filename)[0])
        score_paths.append((path, idx))
    
    score_paths.sort(key=lambda x: x[1])
    
    max_length_of_cluster_id = 0

    buffer = None
    first_index = 0
    max_lengths = {'cluster_id_a': 0, 'cluster_id_b': 0}

    for path, idx in score_paths:
        arr = np.load(path)

        for col in ['cluster_id_a', 'cluster_id_b']:
            max_length = max(arr[col].astype(str), key=len)
            max_lengths[col] = max(max_lengths[col], len(max_length))

        if buffer:
            arr = adjust_string_length(arr, max_lengths)
            buffer = adjust_string_length(buffer, max_lengths)
            arr = np.concatenate((buffer, arr))

        chunk_num = arr.shape[0] // row_size
        for i in range(chunk_num):
            split_arr = arr[i * row_size:(i+1) * row_size]
            split_file_path = os.path.join(dir_path, f'{first_index}.npy')
            np.save(split_file_path, split_arr)
            first_index += row_size

        remainder = arr.shape[0] % row_size
        if remainder:
            buffer = arr[-remainder]
        else:
            buffer = None
    
    if buffer:
        split_file_path = os.path.join(dir_path, f'{first_index}.npy')
        np.save(split_file_path, split_arr)

    # Remove "temp" folder
    shutil.rmtree(temp_folder)


def split_array(arr, row_size):
    if arr.shape[0] <= row_size:
        yield arr

    else:
        chunk_num = arr.shape[0] // row_size
        for i in range(chunk_num):
            split_arr = arr[i * row_size:(i+1) * row_size]
            yield split_arr
        
        remainder = arr.shape[0] % row_size
        yield arr[-remainder:]
    
                
def save_ref_score_file(output_dir):
    ref_clustered_score_dirs = get_paths.get_dirs_of_clustered_score_for_inner_ref()
    ref_clustered_score_dirs += get_paths.get_dirs_of_clustered_score_for_inter_ref()

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for ref_clustered_score_dir in ref_clustered_score_dirs:
        folder_name = os.path.basename(ref_clustered_score_dir)
        dir_to_copy = os.path.join(output_dir, folder_name)
        shutil.copytree(ref_clustered_score_dir, dir_to_copy)

