import h5py

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


def initialize_score_hdf5(path='./score.h5'):
    with h5py.File(path, 'w') as h5:
        pass


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
