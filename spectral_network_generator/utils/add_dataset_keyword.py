import h5py
import numpy as np


def add_dataset_keyword(ref_split_category, path='./spectrum_metadata.h5', key='filtered/metadata'):
    with h5py.File(path, 'a') as h5:
        dset = h5[key]
        arr = dset[()]

        sample_mask = arr['tag'] == b'sample'
        ref_mask = arr['tag'] == b'ref'

        arr['keyword'][sample_mask] = b'sample'

        if ref_split_category == 'cmpd_classification_superclass':
            arr['keyword'][ref_mask] = arr[ref_mask]['cmpd_classification_superclass_list']

        elif ref_split_category == 'cmpd_classification_class':
            arr['keyword'][ref_mask] = arr[ref_mask]['cmpd_classification_class_list']

        elif ref_split_category == 'cmpd_pathway':
            arr['keyword'][ref_mask] = arr[ref_mask]['pathway_common_name_list']

        arr['keyword'][arr['keyword'] == b''] = b'noclassification'

        h5.create_dataset('_metadata', data=dset[()], shape=dset.shape, maxshape=(None,))
        h5.flush()
        del h5[key]
        h5.create_dataset(key, data=arr, shape=arr.shape, maxshape=(None,))
        h5.flush()
        del h5['_metadata']
        h5.flush()


