import h5py
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd

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


if __name__ == '__main__':
    add_cluster_id(score_path='../score.h5')
