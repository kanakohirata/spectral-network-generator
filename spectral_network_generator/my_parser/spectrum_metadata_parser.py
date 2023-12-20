import h5py
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import os
import pandas as pd
import shutil


LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


def get_metadata_dtype():
    dtype = [
        ('index', 'u8'),
        ('tag', 'O'),  # str
        ('keyword', 'O'),  # str
        ('cluster_id', 'O'),  # str
        ('cluster_name', 'O'),  # str
        ('source_filename', 'O'),  # str
        ('global_accession', 'O'),  # str
        ('accession_number', 'O'),  # str
        ('precursor_mz', 'f8'),
        ('rt_in_sec', 'f8'),
        ('retention_index', 'f8'),
        ('inchi', 'O'),  # str
        ('inchikey', 'O'),  # str
        ('author', 'O'),  # str
        ('compound_name', 'O'),  # str
        ('title', 'O'),  # str
        ('instrument_type', 'O'),  # str
        ('ionization_mode', 'O'),  # str
        ('fragmentation_type', 'O'),  # str
        ('precursor_type', 'O'),  # str
        ('number_of_peaks', 'u8'),
        ('peaks', 'O'),  # list
        ('mz_list', 'O'),  # list
        ('external_compound_unique_id_list', 'O'),  # list
        ('pathway_unique_id_list', 'O'),  # list
        ('pathway_common_name_list', 'O'),  # list
        ('cmpd_classification_superclass', 'O'),  # str
        ('cmpd_classification_class', 'O'),  # str
        ('cmpd_classification_subclass', 'O'),  # str
        ('cmpd_classification_alternative_parent_list', 'O')  # list
    ]

    return dtype


def _delete_npy_files(dir_path):
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if os.path.isfile(path) and os.path.splitext(name)[1] == '.npy':
            os.remove(path)


def _delete_folders(dir_path):
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if os.path.isdir(path):
            shutil.rmtree(path)


def initialize_spectrum_metadata_file(dir_path='./spectrum_metadata'):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    else:
        for f in os.listdir(dir_path):
            p = os.path.join(dir_path, f)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.splitext(f)[1] == '.npy':
                os.remove(p)

    dir_path_raw = os.path.join(dir_path, 'raw')
    if not os.path.isdir(dir_path_raw):
        os.makedirs(dir_path_raw)

    dir_path_filtered = os.path.join(dir_path, 'filtered')
    if not os.path.isdir(dir_path_filtered):
        os.makedirs(dir_path_filtered)

    dir_path_grouped = os.path.join(dir_path, 'grouped')
    if not os.path.isdir(dir_path_grouped):
        os.makedirs(dir_path_grouped)


def write_metadata(metadata_path, spectra, export_tsv=False):
    LOGGER.info(f'Write metadata to {os.path.basename(metadata_path)}')

    metadata_dir = os.path.dirname(metadata_path)
    if not os.path.isdir(metadata_dir):
        os.makedirs(metadata_dir)

    metadata_list = []
    dtypes = get_metadata_dtype()
    
    for spectrum in spectra:
        metadata = []
        for dtype in dtypes:
            if dtype[0] in ('index', 'tag', 'source_filename', 'global_accession', 'accession_number',
                            'inchi', 'inchikey', 'author', 'compound_name',
                            'title', 'instrument_type', 'precursor_type',):
                metadata.append(spectrum.get(dtype[0], ''))

            elif dtype[0] in ('keyword', 'cluster_id', 'cluster_name'):
                metadata.append('')

            elif dtype[0] in ('precursor_mz', 'rt_in_sec', 'retention_index'):
                metadata.append(spectrum.get(dtype[0]) or 0)

            elif dtype[0] == 'ionization_mode':
                metadata.append(spectrum.get('ionization_mode')
                                or spectrum.get('ionization')
                                or spectrum.get('ion_mode')
                                or spectrum.get('ionmode', ''))

            elif dtype[0] == 'fragmentation_type':
                metadata.append(spectrum.get('fragmentation_type')
                                or spectrum.get('fragmentation_mode')
                                or spectrum.get('fragmentation', ''))

            elif dtype[0] == 'number_of_peaks':
                metadata.append(spectrum.mz.size)

            elif dtype[0] == 'peaks':
                metadata.append(spectrum.peaks.to_numpy)

            elif dtype[0] == 'mz_list':
                metadata.append(spectrum.mz)

            elif dtype[0] in ('external_compound_unique_id_list', 'pathway_unique_id_list', 'pathway_common_name_list'):
                metadata.append([])

            elif dtype[0] == 'cmpd_classification_superclass':
                metadata.append(spectrum.get('classification_superclass', ''))

            elif dtype[0] == 'cmpd_classification_class':
                metadata.append(spectrum.get('classification_class', ''))

            elif dtype[0] == 'cmpd_classification_subclass':
                metadata.append(spectrum.get('classification_subclass', ''))

            elif dtype[0] == 'cmpd_classification_alternative_parent_list':
                metadata.append([])

        metadata_list.append(tuple(metadata))

    metadata_arr = np.array(metadata_list, dtype=dtypes)

    # Append metadata_arr to an already existing array.
    if os.path.isfile(metadata_path):
        existing_metadata_arr = np.load(metadata_path, allow_pickle=True)
        metadata_arr = np.hstack((existing_metadata_arr, metadata_arr))

    with open(metadata_path, 'wb') as f:
        np.save(f, metadata_arr)

    if export_tsv:
        tsv_path = os.path.splitext(metadata_path)[0] + '.tsv'
        df = pd.DataFrame.from_records(metadata_arr)
        df.to_csv(tsv_path, sep='\t', index=False)


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
