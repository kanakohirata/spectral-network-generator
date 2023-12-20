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


def add_compound_info_old(compound_table_paths):
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
        if 'cmpd_classification_superclass' in df_compound.columns:
            df_compound['cmpd_classification_superclass'].str.encode('utf8')
            df_compound['cmpd_classification_superclass'].fillna(b'', inplace=True)
            df_compound.rename(columns={'cmpd_classification_superclass': 'list_cmpd_classification_superclass_y'}, inplace=True)
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'list_cmpd_classification_superclass_y']],
                                   on='inchikey', how='left')
            
            df_metadata['cmpd_classification_superclass_list_y']\
                = df_metadata['list_cmpd_classification_superclass_y'].apply(
                lambda x: '|'.join(x) if not pd.isna(x) else ''
            )
            df_metadata['cmpd_classification_superclass'] = np.where(
                df_metadata['cmpd_classification_superclass'] == b'',
                df_metadata['cmpd_classification_superclass_list_y'],
                df_metadata['cmpd_classification_superclass']
            )
        else:
            pass

        # Add class
        if 'cmpd_classification_class' in df_compound.columns:
            df_compound['cmpd_classification_class'].str.encode('utf8')
            df_compound['cmpd_classification_class'].fillna(b'', inplace=True)
            df_compound.rename(columns={'cmpd_classification_class': 'cmpd_classification_class_y'}, inplace=True)
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'cmpd_classification_class_y']],
                                   on='inchikey', how='left')
            df_metadata['cmpd_classification_class_y'] \
                = df_metadata['cmpd_classification_class_y'].apply(
                lambda x: '|'.join(x) if not pd.isna(x) else ''
            )
            df_metadata['cmpd_classification_class'] = np.where(
                df_metadata['cmpd_classification_class'] == b'',
                df_metadata['cmpd_classification_class_y'],
                df_metadata['cmpd_classification_class']
            )
        else:
            pass

        # Add subclass
        if 'cmpd_classification_subclass' in df_compound.columns:
            df_compound['cmpd_classification_subclass'].str.encode('utf8')
            df_compound['cmpd_classification_subclass'].fillna(b'', inplace=True)
            df_compound.rename(columns={'cmpd_classification_subclass': 'cmpd_classification_subclass_y'}, inplace=True)
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'cmpd_classification_subclass_y']],
                                   on='inchikey', how='left')
            df_metadata['cmpd_classification_subclass_y'] \
                = df_metadata['cmpd_classification_subclass_y'].apply(
                lambda x: '|'.join(x) if not pd.isna(x) else ''
            )
            df_metadata['cmpd_classification_subclass'] = np.where(
                df_metadata['cmpd_classification_subclass'] == b'',
                df_metadata['cmpd_classification_subclass_y'],
                df_metadata['cmpd_classification_subclass']
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
            ('keyword', H5PY_STR_TYPE), ('cluster_id', H5PY_STR_TYPE),
            ('source_filename', H5PY_STR_TYPE),
            ('global_accession', H5PY_STR_TYPE), ('accession_number', H5PY_STR_TYPE),
            ('precursor_mz', 'f8'), ('rt_in_sec', 'f8'),
            ('retention_index', 'f8'), ('inchi', H5PY_STR_TYPE), ('inchikey', H5PY_STR_TYPE),
            ('author', H5PY_STR_TYPE), ('compound_name', H5PY_STR_TYPE), ('title', H5PY_STR_TYPE),
            ('instrument_type', H5PY_STR_TYPE), ('ionization_mode', H5PY_STR_TYPE),
            ('fragmentation_type', H5PY_STR_TYPE), ('precursor_type', H5PY_STR_TYPE),
            ('number_of_peaks', 'u8'), ('peaks', H5PY_STR_TYPE), ('mz_list', H5PY_STR_TYPE),
            ('external_compound_unique_id_list', H5PY_STR_TYPE), ('pathway_unique_id_list', H5PY_STR_TYPE),
            ('pathway_common_name_list', H5PY_STR_TYPE),
            ('cmpd_classification_superclass', H5PY_STR_TYPE), ('cmpd_classification_class', H5PY_STR_TYPE),
            ('cmpd_classification_subclass', H5PY_STR_TYPE),
            ('cmpd_classification_alternative_parent_list', H5PY_STR_TYPE)
        ])

        h5.create_dataset('filtered/_metadata', data=dset[()], shape=dset.shape, maxshape=(None,))
        h5.flush()
        del h5['filtered/metadata']
        h5.create_dataset('filtered/metadata', data=new_arr, shape=new_arr.shape, maxshape=(None,))
        h5.flush()
        del h5['filtered/_metadata']
        h5.flush()


def add_compound_info(compound_table_paths, metadata_paths):
    df_compound = read_multi_compound_table(compound_table_paths, mode='df')
    if df_compound.empty:
        return

    # Rename columns and fill nan.
    # inchi -> inchi_y
    if 'inchi' in df_compound.columns:
        df_compound['inchi'].fillna('', inplace=True)
        df_compound.rename(columns={'inchi': 'inchi_y'}, inplace=True)
    # name -> compound_name_y
    if 'name' in df_compound.columns:
        df_compound['name'].fillna('', inplace=True)
        df_compound.rename(columns={'name': 'compound_name_y'}, inplace=True)
    # cmpd_classification_superclass -> cmpd_classification_superclass_y
    if 'cmpd_classification_superclass' in df_compound.columns:
        df_compound['cmpd_classification_superclass'].fillna('', inplace=True)
        df_compound.rename(columns={'cmpd_classification_superclass': 'cmpd_classification_superclass_y'}, inplace=True)
    # cmpd_classification_class -> cmpd_classification_class_y
    if 'cmpd_classification_class' in df_compound.columns:
        df_compound['cmpd_classification_class'].fillna('', inplace=True)
        df_compound.rename(columns={'cmpd_classification_class': 'cmpd_classification_class_y'}, inplace=True)
    # cmpd_classification_subclass -> cmpd_classification_subclass_y
    if 'cmpd_classification_subclass' in df_compound.columns:
        df_compound['cmpd_classification_subclass'].fillna('', inplace=True)
        df_compound.rename(columns={'cmpd_classification_subclass': 'cmpd_classification_subclass_y'}, inplace=True)

    for metadata_path in metadata_paths:
        arr = np.load(metadata_path, allow_pickle=True)

        if not arr.size:
            continue

        df_metadata = pd.DataFrame.from_records(arr)

        # Add InChI
        if 'inchi_y' in df_compound.columns:
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'inchi_y']],
                                   on='inchikey', how='left')
            df_metadata.loc[df_metadata['inchi'] == '', 'inchi'] = df_metadata['inchi_y']

        if 'compound_name_y' in df_compound.columns:
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'compound_name_y']],
                                   on='inchikey', how='left')
            df_metadata.loc[df_metadata['compound_name'] == '', 'compound_name'] = df_metadata['compound_name_y']

        # Add superclass
        if 'cmpd_classification_superclass_y' in df_compound.columns:
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'cmpd_classification_superclass_y']],
                                   on='inchikey', how='left')

            df_metadata.loc[df_metadata['cmpd_classification_superclass'] == '', 'cmpd_classification_superclass'] = \
                df_metadata['cmpd_classification_superclass_y']
        else:
            pass

        # Add class
        if 'cmpd_classification_class_y' in df_compound.columns:
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'cmpd_classification_class_y']],
                                   on='inchikey', how='left')

            df_metadata.loc[df_metadata['cmpd_classification_class'] == '', 'cmpd_classification_class'] = \
                df_metadata['cmpd_classification_class_y']
        else:
            pass

        # Add subclass
        if 'cmpd_classification_subclass_y' in df_compound.columns:
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'cmpd_classification_subclass_y']],
                                   on='inchikey', how='left')

            df_metadata.loc[df_metadata['cmpd_classification_subclass'] == '', 'cmpd_classification_subclass'] = \
                df_metadata['cmpd_classification_subclass_y']
        else:
            pass

        df_metadata = df_metadata.loc[:, arr.dtype.names[0]:arr.dtype.names[-1]]

        empty_list_ser = pd.Series([[],] * len(df_metadata))
        empty_ndarray_ser = pd.Series([np.array([]), ] * len(df_metadata))
        for column_name, type_ in df_metadata.dtypes.items():
            print(type(type_))
            if type_ == np.dtype('uint64'):
                df_metadata[column_name].fillna(0, inplace=True)
            elif type_ == np.dtype('float64'):
                df_metadata[column_name].fillna(0.0, inplace=True)
            elif column_name in ('external_compound_unique_id_list',
                                 'pathway_unique_id_list',
                                 'pathway_common_name_list',
                                 'cmpd_classification_alternative_parent_list'):
                df_metadata.loc[pd.isna(df_metadata[column_name]), column_name] = empty_list_ser
            elif column_name in ('peaks', 'mz_list'):
                df_metadata.loc[pd.isna(df_metadata[column_name]), column_name] = empty_ndarray_ser
            else:
                df_metadata[column_name].fillna('', inplace=True)

        new_arr = np.array(df_metadata.to_records(index=False), dtype=arr.dtype)

        np.save(metadata_path, new_arr)
