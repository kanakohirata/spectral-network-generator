from glob import glob
import h5py
import itertools
import logging
from logging import DEBUG, Formatter, getLogger, StreamHandler
from matchms import calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosine
import numpy as np
import numpy.lib.recfunctions as rfn
import os
import pickle
import re
from my_parser.score_parser import get_chunks, initialize_score_hdf5
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
                            similarity_function=CosineGreedy(tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power),
                            is_symmetric=is_symmetric)


def _calculate_similarity_score_with_modified_cosine_greedy(references, queries, tolerance, mz_power=0, intensity_power=1, is_symmetric=False):    
    return calculate_scores(references=references, queries=queries,
                            similarity_function=ModifiedCosine(tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power),
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
    pickle_path_combinations = itertools.combinations_with_replacement(pickle_paths, 2)
    index = 0
    with h5py.File('./score.h5', 'a') as h5, h5py.File('./spectrum_metadata.h5', 'a') as h5_metadata:
        for pickle_path_a, pickle_path_b in pickle_path_combinations:
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
                         f'{os.path.basename(pickle_path_a)} vs {os.path.basename(pickle_path_b)}, intensity conversion: {intensity_convert_message[intensity_convert_mode]}')

            if matching_mode == 1:
                scores = _calculate_similarity_score_with_cosine_greedy(
                    spectra_a, spectra_b, tolerance, intensity_power=intensity_power, is_symmetric=is_symmetric)
            elif matching_mode == 2:
                scores = _calculate_similarity_score_with_modified_cosine_greedy(
                    spectra_a, spectra_b, tolerance, intensity_power=intensity_power, is_symmetric=is_symmetric)

            # Get spectral metadata.
            spectrum_idx_start_a = int(re.findall(r'\d+', os.path.basename(pickle_path_a))[0])
            spectrum_idx_end_a = int(re.findall(r'\d+', os.path.basename(pickle_path_a))[1])
            spectrum_idx_start_b = int(re.findall(r'\d+', os.path.basename(pickle_path_b))[0])
            spectrum_idx_end_b = int(re.findall(r'\d+', os.path.basename(pickle_path_b))[1])

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
                                        ('global_accession_a', H5PY_STR_TYPE), ('global_accession_b', H5PY_STR_TYPE),
                                        ('tag_a', H5PY_STR_TYPE), ('tag_b', H5PY_STR_TYPE),
                                        ('inchikey_a', H5PY_STR_TYPE), ('inchikey_b', H5PY_STR_TYPE),
                                        ('score', 'f8'), ('matches', 'u8')])

            mask = score_arr['index_a'] < score_arr['index_b']
            score_arr = score_arr[mask]
            index_arr = np.arange(index, index + score_arr.shape[0])
            score_arr = rfn.append_fields(score_arr, 'index', index_arr, usemask=False)

            if 'score' in h5.keys():
                dset = h5['score']
                dset.resize((dset.len() + score_arr.shape[0]), axis=0)
                dset[-score_arr.shape[0]:] = score_arr
                h5.flush()
            else:
                h5.create_dataset('score', data=score_arr, shape=score_arr.shape, maxshape=(None,))
                h5.flush()

            index += score_arr.shape[0]


def clustering_based_on_inchikey(chunk_size=1000000):
    last_arr = None
    with h5py.File('./score.h5', 'a') as h5:
        if 'clustered_score' in h5.keys():
            del h5['clustered_score']

        if '_temp' in h5.keys():
            del h5['_temp']

        if '_ref_score' in h5.keys():
            del h5['_ref_score']

        if '_clustered_ref_score' in h5.keys():
            del h5['_clustered_ref_score']
        h5.flush()

        for arr, _start_idx, _end_idx in get_chunks('score', db_chunk_size=chunk_size, path='./score.h5'):
            if last_arr is not None:
                arr = np.hstack((last_arr, arr))

            LOGGER.debug(f'{_start_idx} - {_end_idx}, {arr.shape}')
            idx_arr = arr['index'][(arr['tag_a'] == b'sample') & (arr['tag_b'] == b'sample')]
            sample_vs_ref_mask = (arr['tag_a'] == b'sample') & (arr['tag_b'] == b'ref')
            ref_vs_sample_mask = (arr['tag_a'] == b'ref') & (arr['tag_b'] == b'sample')
            ref_vs_ref_mask = (arr['tag_a'] == b'ref') & (arr['tag_b'] == b'ref')

            if np.any(sample_vs_ref_mask):
                _arr = arr[sample_vs_ref_mask]
                _sorted_indices = np.argsort(_arr, order=['score', 'matches'])[::-1]
                _arr = _arr[_sorted_indices]
                _temp_arr = np.array(_arr[['global_accession_a', 'inchikey_b']].tolist())
                _, _indices = np.unique(_temp_arr, axis=0, return_index=True)
                idx_arr = np.append(idx_arr, _arr['index'][_indices])

            if np.any(ref_vs_sample_mask):
                _arr = arr[ref_vs_sample_mask]
                _sorted_indices = np.argsort(_arr, order=['score', 'matches'])[::-1]
                _arr = _arr[_sorted_indices]
                _temp_arr = np.array(_arr[['global_accession_b', 'inchikey_a']].tolist())
                _, _indices = np.unique(_temp_arr, axis=0, return_index=True)
                idx_arr = np.append(idx_arr, _arr['index'][_indices])

            if np.any(ref_vs_ref_mask):
                _arr = arr[ref_vs_ref_mask]

                if '_ref_score' in h5.keys():
                    _dset = h5['_ref_score']
                    _dset.resize((_dset.len() + _arr.shape[0]), axis=0)
                    _dset[-_arr.shape[0]:] = _arr
                    h5.flush()
                else:
                    h5.create_dataset('_ref_score', data=_arr, shape=_arr.shape, maxshape=(None,))

            idx_arr.sort()
            arr = arr[np.isin(arr['index'], idx_arr)]

            if arr.size:
                last_global_accession_a = arr[-1]['global_accession_a']
                last_global_accession_a_mask = (arr['global_accession_a'] == last_global_accession_a)
                last_arr = arr[last_global_accession_a_mask]

                arr = arr[np.logical_not(last_global_accession_a_mask)]

                if arr.size:
                    if 'clustered_score' in h5.keys():
                        LOGGER.debug('Update clustered_score')
                        dset = h5['clustered_score']
                        dset.resize((dset.len() + arr.shape[0]), axis=0)
                        dset[-arr.shape[0]:] = arr
                        h5.flush()
                    else:
                        LOGGER.debug('Create clustered_score')
                        h5.create_dataset('clustered_score', data=arr, shape=arr.shape, maxshape=(None,))
                        h5.flush()
            else:
                last_arr = None

        if last_arr is not None:
            if last_arr.size:
                dset = h5['clustered_score']
                dset.resize((dset.len() + last_arr.shape[0]), axis=0)
                dset[-last_arr.shape[0]:] = last_arr

        # Cluster scores of ref vs ref
        if '_ref_score' in h5.keys():
            for arr, _start_idx, _end_idx in get_chunks('_ref_score', db_chunk_size=chunk_size, path='./score.h5'):
                LOGGER.debug(f'Ref vs Ref {_start_idx} - {_end_idx}, {arr.shape}')
                if '_clustered_ref_score' in h5.keys():
                    for _clustered_arr, _, _ in get_chunks('_clustered_ref_score', db_chunk_size=chunk_size, path='./score.h5'):
                        _arr = np.hstack((arr, _clustered_arr))
                        _arr = _arr[np.argsort(_arr, order=['score', 'matches'])[::-1]]
                        temp_arr = np.array(_arr[['inchikey_a', 'inchikey_b']].tolist())
                        temp_arr = np.sort(temp_arr, axis=1)
                        _, indices = np.unique(temp_arr, axis=0, return_index=True)
                        arr = arr[np.isin(arr['index'], _arr['index'][indices])]
                        _clustered_arr = _clustered_arr[np.isin(_clustered_arr['index'], _arr['index'][indices])]
                        if '_temp' in h5.keys():
                            _dset = h5['_temp']
                            _dset.resize((_dset.len() + _clustered_arr.shape[0]), axis=0)
                            _dset[-_clustered_arr.shape[0]:] = _clustered_arr
                            h5.flush()
                        else:
                            h5.create_dataset('_temp', data=_clustered_arr, shape=_clustered_arr.shape, maxshape=(None,))
                            h5.flush()

                    _dset = h5['_temp']
                    _clustered_arr = np.hstack((_dset[()], arr))
                    del h5['_temp']
                    del h5['_clustered_ref_score']
                    h5.create_dataset('_clustered_ref_score', data=_clustered_arr,
                                    shape=_clustered_arr.shape, maxshape=(None,))
                    h5.flush()

                else:
                    arr = arr[np.argsort(arr, order=['score', 'matches'])[::-1]]
                    temp_arr = np.array(arr[['inchikey_a', 'inchikey_b']].tolist())
                    temp_arr = np.sort(temp_arr, axis=1)
                    _, indices = np.unique(temp_arr, axis=0, return_index=True)
                    indices.sort()
                    arr = arr[np.isin(arr['index'], arr['index'][indices])]
                    h5.create_dataset('_clustered_ref_score', data=arr, shape=arr.shape, maxshape=(None,))
                    h5.flush()

            # Add clustered scores of ref vs ref
            dset_ref = h5['_clustered_ref_score']
            dset = h5['clustered_score']
            dset.resize((dset.len() + dset_ref.len()), axis=0)
            dset[-dset_ref.len():] = dset_ref[()]

            del h5['_clustered_ref_score']
            del h5['_ref_score']

            h5.flush()


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
