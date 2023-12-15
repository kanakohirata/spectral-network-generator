import h5py
import numpy as np

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


def initialize_spectrum_metadata_hdf5():
    with h5py.File('./spectrum_metadata.h5', 'w') as h5:
        h5.create_group('filtered')


def get_chunks(key, db_chunk_size=10000, path='./spectrum_metadata.h5', change_str_dtype=False):
    """
    Parameters
    ----------
    key: str
        path to dataset
    db_chunk_size: int or None
        Number of compounds loaded at a time
    change_str_dtype: bool
        Whether to change dtype of string to numpy.unicode from numpy.object

    Returns
    -------
    Iterator[numpy.array]
    """

    with h5py.File(path, 'r') as h5:
        dset = h5[key]
        dtype = [('index', 'u8'), ('tag', H5PY_STR_TYPE),
                 ('keyword', H5PY_STR_TYPE), ('cluster_id', H5PY_STR_TYPE),
                 ('source_filename', H5PY_STR_TYPE),
                 ('global_accession', H5PY_STR_TYPE), ('accession_number', H5PY_STR_TYPE),
                 ('precursor_mz', 'f8'), ('rt_in_sec', 'f8'),
                 ('retention_index', 'f8'), ('inchi', H5PY_STR_TYPE), ('inchikey', H5PY_STR_TYPE),
                 ('author', H5PY_STR_TYPE), ('compound_name', H5PY_STR_TYPE), ('title', H5PY_STR_TYPE),
                 ('instrument_type', H5PY_STR_TYPE), ('ionization_mode', H5PY_STR_TYPE),
                 ('fragmentation_type', H5PY_STR_TYPE), ('precursor_type', H5PY_STR_TYPE),
                 ('number_of_peaks', 'u8'), ('peaks', H5PY_STR_TYPE), ('mz_list', H5PY_STR_TYPE),
                 ('external_compound_unique_id_list', H5PY_STR_TYPE),
                 ('pathway_unique_id_list', H5PY_STR_TYPE),
                 ('pathway_common_name_list', H5PY_STR_TYPE),
                 ('cmpd_classification_superclass_list', H5PY_STR_TYPE),
                 ('cmpd_classification_class_list', H5PY_STR_TYPE),
                 ('cmpd_classification_subclass_list', H5PY_STR_TYPE),
                 ('cmpd_classification_alternative_parent_list', H5PY_STR_TYPE)]

        if db_chunk_size:
            chunk_num = int(dset.size / db_chunk_size) + 1

        else:
            db_chunk_size = dset.size
            chunk_num = 1

        for i in range(chunk_num):
            start = db_chunk_size * i
            end = db_chunk_size * (i + 1)

            if change_str_dtype:
                yield dset[start:end].astype(dtype, copy=False), start, end  # Too late
            else:
                yield dset[start:end], start, end


def group_by_dataset(ref_split_category, path='./spectrum_metadata.h5', key='filtered/metadata'):
    with h5py.File(path, 'a') as h5:
        if 'grouped' in h5.keys():
            del h5['grouped']

        dset = h5[key]
        if ref_split_category in ('cmpd_classification_superclass', 'cmpd_classification_class'):
            keywords_arr = np.array(list(map(lambda x: x.split(b'|'), dset['keyword'])))
        else:
            keywords_arr = np.array(list(map(lambda x: x.split(b', '), dset['keyword'])))
        keyword_set = []
        for keywords in keywords_arr:
            keyword_set.extend(keywords)

        keyword_set = set(keyword_set)

        for dataset_keyword in keyword_set:
            dataset_mask = [dataset_keyword in k for k in keywords_arr]
            dataset_arr = dset[()][dataset_mask]
            dataset_arr['keyword'] = dataset_keyword
            h5.create_dataset(f'grouped/{dataset_keyword.decode("utf-8")}', data=dataset_arr, shape=dataset_arr.shape, maxshape=(None,))
