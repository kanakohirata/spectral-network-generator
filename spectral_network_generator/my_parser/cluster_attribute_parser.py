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


def write_cluster_attribute(output_path, metadata_path):
    """
    Parameters
    ----------
    output_path : str
        Path of cluster attribute file.
    metadata_path : str
        Path of metadata file.
    Returns
    -------

    """
    LOGGER.debug(f'Write cluster attribute: {output_path}')

    # Load metadata array
    metadata_arr = np.load(metadata_path, allow_pickle=True)
    df_meta = pd.DataFrame.from_records(metadata_arr)
    del metadata_arr

    df_meta = df_meta[[
        'index', 'cluster_id', 'keyword', 'global_accession', 'tag',
        'inchi', 'inchikey', 'accession_number', 'compound_name', 'source_filename',
        'external_compound_unique_id_list', 'pathway_unique_id_list', 'pathway_common_name_list',
        'rt_in_sec', 'precursor_mz', 'peaks',
        'cmpd_classification_superclass', 'cmpd_classification_class',
        'cmpd_classification_alternative_parent_list'
    ]]

    # Change types
    df_meta['peaks'] = df_meta['peaks'].apply(lambda peaks: peaks.tolist())

    # Add 'cmpd_classification_kingdom' column
    df_meta['cmpd_classification_kingdom'] = \
        df_meta['cmpd_classification_superclass'].apply(
            lambda superclass: 'noclassification'
            if superclass in ('noclassification', '')
            else (
                'Inorganic compounds'
                if superclass in ('Homogeneous non-metal compounds', 'Mixed metal/non-metal compounds')
                else 'Organic compounds'
            )
        )

    # Change type of classification from str to list ----------------------------
    # !!! This changing is temporary for spectral network visualizer.
    #     and will be removed in the future.
    df_meta['cmpd_classification_kingdom'] = df_meta['cmpd_classification_kingdom'].apply(lambda class_: [class_, ])
    df_meta['cmpd_classification_superclass'] =\
        df_meta['cmpd_classification_superclass'].apply(lambda class_: [class_, ])
    df_meta['cmpd_classification_class'] = df_meta['cmpd_classification_class'].apply(lambda class_: [class_, ])
    # ---------------------------------------------------------------------------

    # Count spectra with the same cluster_id and create list of them accession numbers --------------------
    df_meta['NUMBER_OF_SPECTRA'] = 1
    df_meta['LIST_ACCESSION_NUMBERS'] = df_meta['accession_number']

    df_cluster_id_count =\
        df_meta[['cluster_id', 'NUMBER_OF_SPECTRA', 'LIST_ACCESSION_NUMBERS']].groupby('cluster_id').agg(
            {'NUMBER_OF_SPECTRA': 'sum',
             'LIST_ACCESSION_NUMBERS': lambda x: ', '.join(x)}
        )
    df_cluster_id_count['cluster_id'] = df_cluster_id_count.index
    df_cluster_id_count.reset_index(drop=True, inplace=True)
    # -----------------------------------------------------------------------------------------------------

    # Retain unique cluster_id
    df_meta_unique = df_meta.drop_duplicates(subset='cluster_id')
    df_meta_unique = pd.merge(df_meta_unique, df_cluster_id_count, on='cluster_id', how='left')
    df_meta_unique['NUMBER_OF_SPECTRA'] = df_meta_unique['NUMBER_OF_SPECTRA_y']
    df_meta_unique['LIST_ACCESSION_NUMBERS'] = df_meta_unique['LIST_ACCESSION_NUMBERS_y']

    # 'UNIQUE_LEVEL', 'PEAK_LIST_MZ_ANNOID' and 'DIC_ANNOID_STRUCT' are not working.
    df_meta_unique['UNIQUE_LEVEL'] = -1
    df_meta_unique['PEAK_LIST_MZ_ANNOID'] = '[]'
    df_meta_unique['DIC_ANNOID_STRUCT'] = '{}'

    df_meta_unique.rename(inplace=True,
                          columns={
                              'cluster_id': 'CLUSTER_ID',
                              'global_accession': 'GLOBAL_ACCESSION',
                              'tag': 'DATASET',
                              'keyword': 'DS_SPLIT_KEY',
                              'inchi': 'INCHI',
                              'inchikey': 'INCHI_KEY',
                              'accession_number': 'ACCESSION_NUMBER',
                              'compound_name': 'CMPD_NAME',
                              'source_filename': 'FILE_NAME',
                              'external_compound_unique_id_list': 'PATHWAY_UNIQUE_ID',
                              'pathway_unique_id_list': 'PATHWAY_COMMON_NAME',
                              'pathway_common_name_list': 'PATHWAY_COMMON_NAMEX',
                              'rt_in_sec': 'RETENTION_TIME_IN_SEC',
                              'precursor_mz': 'PRECURSOR_MZ',
                              'peaks': 'PEAK_LIST_MZ_INT_REL',
                              'cmpd_classification_kingdom': 'CMPD_CLASSIFICATION_KINGDOM',
                              'cmpd_classification_superclass': 'CMPD_CLASSIFICATION_SUPERCLASS',
                              'cmpd_classification_class': 'CMPD_CLASSIFICATION_CLASS',
                              'cmpd_classification_alternative_parent_list': 'CMPD_CLASSIFICATION_ALTERNATIVEPARENT'
                          })

    df_meta_unique = df_meta_unique[[
        'CLUSTER_ID', 'GLOBAL_ACCESSION', 'DATASET', 'DS_SPLIT_KEY', 'INCHI', 'INCHI_KEY', 'ACCESSION_NUMBER',
        'CMPD_NAME', 'FILE_NAME', 'PATHWAY_UNIQUE_ID', 'PATHWAY_COMMON_NAME', 'PATHWAY_COMMON_NAMEX',
        'RETENTION_TIME_IN_SEC', 'PRECURSOR_MZ', 'UNIQUE_LEVEL', 'PEAK_LIST_MZ_INT_REL', 'PEAK_LIST_MZ_ANNOID',
        'DIC_ANNOID_STRUCT', 'NUMBER_OF_SPECTRA', 'LIST_ACCESSION_NUMBERS', 'CMPD_CLASSIFICATION_KINGDOM',
        'CMPD_CLASSIFICATION_SUPERCLASS', 'CMPD_CLASSIFICATION_CLASS', 'CMPD_CLASSIFICATION_ALTERNATIVEPARENT'
    ]]

    df_meta_unique.to_csv(output_path, sep='\t', index=False)

