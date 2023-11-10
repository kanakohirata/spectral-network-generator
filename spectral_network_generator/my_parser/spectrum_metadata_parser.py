import h5py

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
        dtype = [('tag', 'u8'), ('tag', H5PY_STR_TYPE), ('source_filename', H5PY_STR_TYPE),
                 ('accession_number', H5PY_STR_TYPE), ('precursor_mz', 'f8'), ('rt_in_sec', 'f8'),
                 ('retention_index', 'f8'), ('inchi', H5PY_STR_TYPE), ('inchikey', H5PY_STR_TYPE),
                 ('author', H5PY_STR_TYPE), ('compound_name', H5PY_STR_TYPE), ('instrument_type', H5PY_STR_TYPE),
                 ('ionization_mode', H5PY_STR_TYPE), ('fragmentation_type', H5PY_STR_TYPE),
                 ('precursor_type', H5PY_STR_TYPE), ('number_of_peaks', 'u8')]

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
