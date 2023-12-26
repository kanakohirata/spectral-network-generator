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
        df_compound['cmpd_classification_superclass'].fillna('noclassification', inplace=True)
        df_compound.rename(columns={'cmpd_classification_superclass': 'cmpd_classification_superclass_y'}, inplace=True)
    # cmpd_classification_class -> cmpd_classification_class_y
    if 'cmpd_classification_class' in df_compound.columns:
        df_compound['cmpd_classification_class'].fillna('noclassification', inplace=True)
        df_compound.rename(columns={'cmpd_classification_class': 'cmpd_classification_class_y'}, inplace=True)
    # cmpd_classification_subclass -> cmpd_classification_subclass_y
    if 'cmpd_classification_subclass' in df_compound.columns:
        df_compound['cmpd_classification_subclass'].fillna('noclassification', inplace=True)
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
            # Fill nan
            df_metadata[['inchi', 'inchi_y']] = df_metadata[['inchi', 'inchi_y']].fillna('')
            df_metadata.loc[df_metadata['inchi'] == '', 'inchi'] = df_metadata['inchi_y']

        if 'compound_name_y' in df_compound.columns:
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'compound_name_y']],
                                   on='inchikey', how='left')
            # Fill nan
            df_metadata[['compound_name', 'compound_name_y']] = df_metadata[['compound_name', 'compound_name_y']].fillna('')
            df_metadata.loc[df_metadata['compound_name'] == '', 'compound_name'] = df_metadata['compound_name_y']

        # Add superclass
        if 'cmpd_classification_superclass_y' in df_compound.columns:
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'cmpd_classification_superclass_y']],
                                   on='inchikey', how='left')
            # Fill nan
            df_metadata['cmpd_classification_superclass_y'].fillna('noclassification', inplace=True)

            df_metadata.loc[df_metadata['cmpd_classification_superclass'] == 'noclassification',
                            'cmpd_classification_superclass'] = df_metadata['cmpd_classification_superclass_y']
        else:
            pass

        # Add class
        if 'cmpd_classification_class_y' in df_compound.columns:
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'cmpd_classification_class_y']],
                                   on='inchikey', how='left')
            df_metadata['cmpd_classification_class_y'].fillna('noclassification', inplace=True)

            df_metadata.loc[df_metadata['cmpd_classification_class'] == 'noclassification',
                            'cmpd_classification_class'] = df_metadata['cmpd_classification_class_y']
        else:
            pass

        # Add subclass
        if 'cmpd_classification_subclass_y' in df_compound.columns:
            df_metadata = pd.merge(df_metadata, df_compound[['inchikey', 'cmpd_classification_subclass_y']],
                                   on='inchikey', how='left')
            df_metadata['cmpd_classification_subclass_y'].fillna('noclassification', inplace=True)

            df_metadata.loc[df_metadata['cmpd_classification_subclass'] == 'noclassification',
                            'cmpd_classification_subclass'] = df_metadata['cmpd_classification_subclass_y']
        else:
            pass

        df_metadata = df_metadata.loc[:, arr.dtype.names[0]:arr.dtype.names[-1]]

        empty_list_ser = pd.Series([[],] * len(df_metadata))
        empty_ndarray_ser = pd.Series([np.array([]), ] * len(df_metadata))
        for column_name, type_ in df_metadata.dtypes.items():
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

        with open(metadata_path, 'wb') as f:
            np.save(f, new_arr)
            f.flush()
