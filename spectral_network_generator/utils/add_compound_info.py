import h5py
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import pandas as pd
from my_parser.compound_table_parser import read_multi_compound_table

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


def add_compound_info(compound_table_paths):
    df_compound = read_multi_compound_table(compound_table_paths, mode='df')
    if df_compound.empty:
        return

    with h5py.File('./spectrum_metadata.h5', 'a') as h5:
        dset = h5['filtered/metadata']
        df_metadata = pd.DataFrame.from_records(dset[()])
        df_metadata['inchikey'] = df_metadata['inchikey'].str.decode('utf8')

        # Add InChI
        if 'inchi' in df_compound.columns:
            df_compound['inchi'] = df_compound['inchi'].str.encode('utf8')
            df_compound['inchi'].fillna(b'', inplace=True)
            df_compound.rename(columns={'inchi': 'inchi_y'}, inplace=True)
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'inchi_y']],
                                   on='inchikey', how='left')
            df_metadata['inchi'] = np.where(df_metadata['inchi'] == b'', df_metadata['inchi_y'], df_metadata['inchi'])

        if 'name' in df_compound.columns:
            df_compound['name'].str.encode('utf8')
            df_compound['name'].fillna(b'', inplace=True)
            df_compound.rename(columns={'name': 'compound_name_y'}, inplace=True)
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'compound_name_y']],
                                   on='inchikey', how='left')
            df_metadata['compound_name'] = np.where(df_metadata['compound_name'] == b'', df_metadata['compound_name_y'], df_metadata['compound_name'])
        
        # Add superclass
        if 'list_cmpd_classification_superclass' in df_compound.columns:
            df_compound['list_cmpd_classification_superclass'].str.encode('utf8')
            df_compound['list_cmpd_classification_superclass'].fillna(b'', inplace=True)
            df_compound.rename(columns={'list_cmpd_classification_superclass': 'list_cmpd_classification_superclass_y'}, inplace=True)
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'list_cmpd_classification_superclass_y']],
                                   on='inchikey', how='left')
            
            df_metadata['cmpd_classification_superclass_list_y']\
                = df_metadata['list_cmpd_classification_superclass_y'].apply(
                lambda x: '|'.join(x) if not pd.isna(x) else ''
            )
            df_metadata['cmpd_classification_superclass_list'] = np.where(
                df_metadata['cmpd_classification_superclass_list'] == b'',
                df_metadata['cmpd_classification_superclass_list_y'],
                df_metadata['cmpd_classification_superclass_list']
            )
        else:
            pass

        # Add class
        if 'list_cmpd_classification_class' in df_compound.columns:
            df_compound['list_cmpd_classification_class'].str.encode('utf8')
            df_compound['list_cmpd_classification_class'].fillna(b'', inplace=True)
            df_compound.rename(columns={'list_cmpd_classification_class': 'list_cmpd_classification_class_y'}, inplace=True)
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'list_cmpd_classification_class_y']],
                                   on='inchikey', how='left')
            df_metadata['cmpd_classification_class_list_y'] \
                = df_metadata['list_cmpd_classification_class_y'].apply(
                lambda x: '|'.join(x) if not pd.isna(x) else ''
            )
            df_metadata['cmpd_classification_class_list'] = np.where(
                df_metadata['cmpd_classification_class_list'] == b'',
                df_metadata['cmpd_classification_class_list_y'],
                df_metadata['cmpd_classification_class_list']
            )
        else:
            pass

        # Add subclass
        if 'list_cmpd_classification_subclass' in df_compound.columns:
            df_compound['list_cmpd_classification_subclass'].str.encode('utf8')
            df_compound['list_cmpd_classification_subclass'].fillna(b'', inplace=True)
            df_compound.rename(columns={'list_cmpd_classification_subclass': 'list_cmpd_classification_subclass_y'}, inplace=True)
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'list_cmpd_classification_subclass_y']],
                                   on='inchikey', how='left')
            df_metadata['cmpd_classification_subclass_list_y'] \
                = df_metadata['list_cmpd_classification_subclass_y'].apply(
                lambda x: '|'.join(x) if not pd.isna(x) else ''
            )
            df_metadata['cmpd_classification_subclass_list'] = np.where(
                df_metadata['cmpd_classification_subclass_list'] == b'',
                df_metadata['cmpd_classification_subclass_list_y'],
                df_metadata['cmpd_classification_subclass_list']
            )
        else:
            pass

        df_metadata = df_metadata.loc[:, 'index':'cmpd_classification_alternative_parent_list']

        for column_name, type_ in df_metadata.dtypes.items():
            print(type(type_))
            if type_ == np.dtype('object_'):
                df_metadata[column_name].fillna(b'', inplace=True)
            elif type_ == np.dtype('uint64'):
                df_metadata[column_name].fillna(0, inplace=True)
            elif type_ == np.dtype('float64'):
                df_metadata[column_name].fillna(0.0, inplace=True)

        new_arr = np.array(df_metadata.to_records(index=False), dtype=[
            ('index', 'u8'), ('tag', H5PY_STR_TYPE),
            ('keyword', H5PY_STR_TYPE), ('source_filename', H5PY_STR_TYPE),
            ('global_accession', H5PY_STR_TYPE), ('accession_number', H5PY_STR_TYPE),
            ('precursor_mz', 'f8'), ('rt_in_sec', 'f8'),
            ('retention_index', 'f8'), ('inchi', H5PY_STR_TYPE), ('inchikey', H5PY_STR_TYPE),
            ('author', H5PY_STR_TYPE), ('compound_name', H5PY_STR_TYPE), ('title', H5PY_STR_TYPE),
            ('instrument_type', H5PY_STR_TYPE), ('ionization_mode', H5PY_STR_TYPE),
            ('fragmentation_type', H5PY_STR_TYPE), ('precursor_type', H5PY_STR_TYPE),
            ('number_of_peaks', 'u8'), ('peaks', H5PY_STR_TYPE), ('mz_list', H5PY_STR_TYPE),
            ('external_compound_unique_id_list', H5PY_STR_TYPE), ('pathway_unique_id_list', H5PY_STR_TYPE),
            ('pathway_common_name_list', H5PY_STR_TYPE),
            ('cmpd_classification_superclass_list', H5PY_STR_TYPE), ('cmpd_classification_class_list', H5PY_STR_TYPE),
            ('cmpd_classification_subclass_list', H5PY_STR_TYPE),
            ('cmpd_classification_alternative_parent_list', H5PY_STR_TYPE)
        ])

        h5.create_dataset('filtered/_metadata', data=dset[()], shape=dset.shape, maxshape=(None,))
        h5.flush()
        del h5['filtered/metadata']
        h5.create_dataset('filtered/metadata', data=new_arr, shape=new_arr.shape, maxshape=(None,))
        h5.flush()
        del h5['filtered/_metadata']
        h5.flush()
