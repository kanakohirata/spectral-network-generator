import h5py
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
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


def add_metacyc_compound_info():
    LOGGER.debug('Add metacyc compound info to metadata.')
    with h5py.File('./metacyc.h5', 'r') as h5_metacyc, h5py.File('./spectrum_metadata.h5', 'a') as h5_metadata:
        dset_metadata = h5_metadata['filtered/metadata']
        dset_metacyc_compound = h5_metacyc['compound']

        if not dset_metacyc_compound.size:
            return

        df_metadata = pd.DataFrame.from_records(dset_metadata[()])
        df_metacyc_compound = pd.DataFrame.from_records(dset_metacyc_compound[()])

        df_metacyc_compound['unique_id'] = df_metacyc_compound['unique_id'].str.decode('utf-8')
        df_metacyc_compound['pathway_unique_id_list'] = df_metacyc_compound['pathway_unique_id_list'].str.decode('utf-8')
        df_metacyc_compound['pathway_common_name_list'] = df_metacyc_compound['pathway_common_name_list'].str.decode('utf-8')

        df_metadata['inchikey_head'] = df_metadata['inchikey'].str.decode('utf-8').str.split('-', expand=True)[0]
        df_metacyc_compound['inchikey_head'] = df_metacyc_compound['inchikey'].str.decode('utf-8').str.split('-', expand=True)[0]

        df_metadata[['external_compound_unique_id_list', 'pathway_unique_id_list', 'pathway_common_name_list']] = ['', '', '']

        for _i, row in df_metadata[df_metadata['inchikey_head'] != ''].iterrows():
            mask = df_metacyc_compound['inchikey_head'].apply(lambda x: row['inchikey_head'] == x)
            str_external_compound_unique_ids = ', '.join(df_metacyc_compound[mask]['unique_id'])

            pathway_unique_ids = tuple(i for i in df_metacyc_compound[mask]['pathway_unique_id_list'] if i)
            pathway_common_names = tuple(n for n in df_metacyc_compound[mask]['pathway_common_name_list'] if n)
            str_pathway_unique_ids = ', '.join(pathway_unique_ids)
            str_pathway_common_names = ', '.join(pathway_common_names)

            df_metadata.loc[_i, 'external_compound_unique_id_list':'pathway_common_name_list'] = [
                str_external_compound_unique_ids, str_pathway_unique_ids, str_pathway_common_names
            ]

        new_arr = np.array(
            df_metadata.loc[:, :'cmpd_classification_alternative_parent_list'].to_records(index=False),
            dtype=[
                ('index', 'u8'), ('tag', H5PY_STR_TYPE), ('source_filename', H5PY_STR_TYPE),
                ('global_accession', H5PY_STR_TYPE), ('accession_number', H5PY_STR_TYPE),
                ('precursor_mz', 'f8'), ('rt_in_sec', 'f8'),
                ('retention_index', 'f8'), ('inchi', H5PY_STR_TYPE), ('inchikey', H5PY_STR_TYPE),
                ('author', H5PY_STR_TYPE), ('compound_name', H5PY_STR_TYPE), ('title', H5PY_STR_TYPE),
                ('instrument_type', H5PY_STR_TYPE), ('ionization_mode', H5PY_STR_TYPE),
                ('fragmentation_type', H5PY_STR_TYPE), ('precursor_type', H5PY_STR_TYPE),
                ('number_of_peaks', 'u8'), ('peaks', H5PY_STR_TYPE), ('mz_list', H5PY_STR_TYPE),
                ('external_compound_unique_id_list', H5PY_STR_TYPE), ('pathway_unique_id_list', H5PY_STR_TYPE),
                ('pathway_common_name_list', H5PY_STR_TYPE),
                ('cmpd_classification_superclass_list', H5PY_STR_TYPE),
                ('cmpd_classification_class_list', H5PY_STR_TYPE),
                ('cmpd_classification_subclass_list', H5PY_STR_TYPE),
                ('cmpd_classification_alternative_parent_list', H5PY_STR_TYPE)
            ]
        )

        h5_metadata.create_dataset('filtered/_metadata', data=dset_metadata[()],
                                   shape=dset_metadata.shape, maxshape=(None,))
        h5_metadata.flush()
        del h5_metadata['filtered/metadata']
        h5_metadata.create_dataset('filtered/metadata', data=new_arr,
                                   shape=new_arr.shape, maxshape=(None,))
        h5_metadata.flush()
        del h5_metadata['filtered/_metadata']
        h5_metadata.flush()

