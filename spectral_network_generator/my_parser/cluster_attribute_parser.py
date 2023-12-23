import h5py
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import pandas as pd
import re
from my_parser.score_parser import iter_clustered_score_array

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


def write_cluster_attribute_old(path, spectra_datasets):
    data = []
    for sp_ds in spectra_datasets:
        for s in sp_ds.matchms_spectra:
            data.append([
                s.metadata['cluster_id'], s.metadata['represen_spec_global_accession'], sp_ds.tag, sp_ds.keyword,
                s.get('inchi'), s.get('inchikey'), s.metadata['accession_number'], s.get('name'),
                s.metadata['source_filename'], s.get('list_external_compound_UNIQUE_ID', '[]'),
                s.get('list_pathway_UNIQUE_ID', '[]'), s.get('list_pathway_COMMON_NAME', '[]'),
                s.get('retention_time_in_sec', 0), s.get('precursor_mz', 100), s.get('unique_level', -1),
                s.peaks.to_numpy.tolist(), s.get('peak_list_mz_annoid', '[]'), s.get('dic_annoid_struct', '{}'), 1,
                s.metadata['accession_number'], s.get('list_cmpd_classification_kingdom', "['noclassification']"),
                s.get('list_cmpd_classification_class', "['noclassification']"),
                s.get('list_cmpd_classification_superclass', "['noclassification']"),
                s.get('list_cmpd_classification_alternative_parent', "['noclassification']"),
            ])

    df = pd.DataFrame(data, columns=[
        'CLUSTER_ID', 'GLOBAL_ACCESSION', 'DATASET', 'DS_SPLIT_KEY', 'INCHI', 'INCHI_KEY', 'ACCESSION_NUMBER',
        'CMPD_NAME', 'FILE_NAME', 'PATHWAY_UNIQUE_ID', 'PATHWAY_COMMON_NAME', 'PATHWAY_COMMON_NAMEX',
        'RETENTION_TIME_IN_SEC', 'PRECURSOR_MZ', 'UNIQUE_LEVEL', 'PEAK_LIST_MZ_INT_REL', 'PEAK_LIST_MZ_ANNOID',
        'DIC_ANNOID_STRUCT', 'NUMBER_OF_SPECTRA', 'LIST_ACCESSION_NUMBERS', 'CMPD_CLASSIFICATION_KINGDOM',
        'CMPD_CLASSIFICATION_CLASS', 'CMPD_CLASSIFICATION_SUPERCLASS', 'CMPD_CLASSIFICATION_ALTERNATIVEPARENT'
    ])

    df.to_csv(path, sep='\t', index=False)


def write_cluster_attribute(path, ref_split_category):
    LOGGER.debug(f'Write cluster attribute: {path}')
    arr = None
    for _arr, chunk_start, chunk_end in get_chunks('clustered_score', db_chunk_size=1000000, path='./score.h5'):
        _arr_a = _arr[['global_accession_a', 'cluster_id_a']]
        _arr_b = _arr[['global_accession_b', 'cluster_id_b']]

        _arr_a = np.unique(_arr_a)
        _arr_b = np.unique(_arr_b)

        _arr_a = _arr_a.astype([('global_accession', 'O'), ('cluster_id', 'O')])
        _arr_b = _arr_b.astype([('global_accession', 'O'), ('cluster_id', 'O')])

        _arr = np.hstack((_arr_a, _arr_b))

        if arr is None:
            arr = np.unique(_arr)
        else:
            _arr = np.hstack((arr, _arr))
            arr = np.unique(_arr)

    df_cluster_id = pd.DataFrame.from_records(arr)
    df_cluster_id['global_accession'] = df_cluster_id['global_accession'].str.decode('utf-8')
    df_cluster_id['cluster_id'] = df_cluster_id['cluster_id'].str.decode('utf-8')

    with h5py.File('./spectrum_metadata.h5', 'r') as h5_meta:
        dset_meta = h5_meta['filtered/metadata']
        df_meta = pd.DataFrame.from_records(dset_meta[()][[
            'index', 'global_accession', 'tag', 'inchi', 'inchikey', 'accession_number', 'compound_name', 'source_filename',
            'external_compound_unique_id_list', 'pathway_unique_id_list', 'pathway_common_name_list',
            'rt_in_sec', 'precursor_mz', 'peaks',
            'cmpd_classification_superclass_list', 'cmpd_classification_class_list',
            'cmpd_classification_alternative_parent_list'
        ]])

        # Change types
        df_meta['global_accession'] = df_meta['global_accession'].str.decode('utf-8')
        df_meta['tag'] = df_meta['tag'].str.decode('utf-8')
        df_meta['inchi'] = df_meta['inchi'].str.decode('utf-8')
        df_meta['inchikey'] = df_meta['inchikey'].str.decode('utf-8')
        df_meta['accession_number'] = df_meta['accession_number'].str.decode('utf-8')
        df_meta['compound_name'] = df_meta['compound_name'].str.decode('utf-8')
        df_meta['source_filename'] = df_meta['source_filename'].str.decode('utf-8')
        df_meta['external_compound_unique_id_list'] = df_meta['external_compound_unique_id_list'].str.decode('utf-8')
        df_meta['pathway_unique_id_list'] = df_meta['pathway_unique_id_list'].str.decode('utf-8')
        df_meta['pathway_common_name_list'] = df_meta['pathway_common_name_list'].str.decode('utf-8')
        df_meta['peaks'] = df_meta['peaks'].str.decode('utf-8')
        df_meta['cmpd_classification_superclass_list'] = df_meta['cmpd_classification_superclass_list'].str.decode('utf-8')
        df_meta['cmpd_classification_class_list'] = df_meta['cmpd_classification_class_list'].str.decode('utf-8')
        df_meta['cmpd_classification_alternative_parent_list'] =\
            df_meta['cmpd_classification_alternative_parent_list'].str.decode('utf-8')

        # Convert pathway unique id from str to np.array
        df_meta['external_compound_unique_id_list'] = df_meta['external_compound_unique_id_list'].apply(
            lambda x: re.findall(r'[A-Za-z0-9-]+', x))

        # Convert pathway unique id from str to np.array
        df_meta['pathway_unique_id_list'] = df_meta['pathway_unique_id_list'].apply(
            lambda x: re.findall(r'[A-Za-z0-9-]+', x))

        # Convert pathway common name from str to np.array
        df_meta['pathway_common_name_list'] = df_meta['pathway_common_name_list'].apply(
            lambda x: x.split(', ') if x else [])

        # Convert compound classes from str to list
        df_meta['cmpd_classification_superclass_list'] = \
            df_meta['cmpd_classification_superclass_list'].apply(lambda x: x.split('|') if x else [])
        df_meta['cmpd_classification_class_list'] = \
            df_meta['cmpd_classification_class_list'].apply(lambda x: x.split('|') if x else [])
        df_meta['cmpd_classification_alternative_parent_list'] = \
            df_meta['cmpd_classification_alternative_parent_list'].apply(lambda x: x.split('|') if x else [])

        # Add 'CMPD_CLASSIFICATION_KINGDOM' column
        df_meta['CMPD_CLASSIFICATION_KINGDOM'] = \
            df_meta['cmpd_classification_superclass_list'].apply(
                lambda sl: ['Inorganic compounds'
                            if s in ('Homogeneous non-metal compounds', 'Mixed metal/non-metal compounds')
                            else 'Organic compounds' for s in sl])

        df_meta = pd.merge(df_meta, df_cluster_id, on='global_accession', how='left')

        # Fill empty "cluster_id" of sample
        df_meta.loc[(df_meta['tag'] == 'sample') & (pd.isna(df_meta['cluster_id'])), 'cluster_id'] = df_meta['global_accession']

        # Fill empty "cluster_id" of reference
        for idx, row in df_meta[(df_meta['tag'] == 'ref') & (pd.isna(df_meta['cluster_id']))].iterrows():
            if row['inchikey']:
                _df_same_inchikey = df_meta[(df_meta['tag'] == 'ref') & (df_meta['inchikey'] == row['inchikey']) & (~pd.isna(df_meta['cluster_id']))]

                if len(_df_same_inchikey):
                    df_meta.at[idx, 'cluster_id'] = _df_same_inchikey.iloc[0]['cluster_id']

        df_meta['NUMBER_OF_SPECTRA'] = 1
        df_meta['LIST_ACCESSION_NUMBERS'] = df_meta['accession_number']

        df_cluster_id_count = df_meta[['cluster_id', 'NUMBER_OF_SPECTRA', 'LIST_ACCESSION_NUMBERS']].groupby('cluster_id').agg(
            {'NUMBER_OF_SPECTRA': 'sum',
             'LIST_ACCESSION_NUMBERS': lambda x: ', '.join(x)}
        )
        df_cluster_id_count['cluster_id'] = df_cluster_id_count.index
        df_cluster_id_count.reset_index(drop=True, inplace=True)
        df_meta_unique = df_meta.drop_duplicates(subset='cluster_id')
        df_meta_unique = df_meta_unique.loc[:, :'cluster_id']
        df_meta_unique = pd.merge(df_meta_unique, df_cluster_id_count, on='cluster_id', how='left')

        if ref_split_category == 'cmpd_classification_superclass':
            df_meta_unique['DS_SPLIT_KEY'] = df_meta_unique['cmpd_classification_superclass_list'].apply(lambda s: s[0] if s else '')
        elif ref_split_category == 'cmpd_classification_class':
            df_meta_unique['DS_SPLIT_KEY'] = df_meta_unique['cmpd_classification_class_list'].apply(lambda c: c[0] if c else '')
        elif ref_split_category == 'cmpd_pathway':
            df_meta_unique['DS_SPLIT_KEY'] = df_meta_unique['pathway_unique_id_list'].apply(lambda p: p[0] if p else '')
        else:
            df_meta_unique['DS_SPLIT_KEY'] = ''

        # 'UNIQUE_LEVEL', 'PEAK_LIST_MZ_ANNOID' and 'DIC_ANNOID_STRUCT' are not working.
        df_meta_unique['UNIQUE_LEVEL'] = -1
        df_meta_unique['PEAK_LIST_MZ_ANNOID'] = '[]'
        df_meta_unique['DIC_ANNOID_STRUCT'] = '{}'

        df_meta_unique.rename(inplace=True,
                              columns={
                                  'cluster_id': 'CLUSTER_ID', 'global_accession': 'GLOBAL_ACCESSION',
                                  'tag': 'DATASET', 'inchi': 'INCHI', 'inchikey': 'INCHI_KEY',
                                  'accession_number': 'ACCESSION_NUMBER', 'compound_name': 'CMPD_NAME',
                                  'source_filename': 'FILE_NAME',
                                  'external_compound_unique_id_list': 'PATHWAY_UNIQUE_ID',
                                  'pathway_unique_id_list': 'PATHWAY_COMMON_NAME',
                                  'pathway_common_name_list': 'PATHWAY_COMMON_NAMEX',
                                  'rt_in_sec': 'RETENTION_TIME_IN_SEC', 'precursor_mz': 'PRECURSOR_MZ',
                                  'peaks': 'PEAK_LIST_MZ_INT_REL',
                                  'cmpd_classification_superclass_list': 'CMPD_CLASSIFICATION_SUPERCLASS',
                                  'cmpd_classification_class_list': 'CMPD_CLASSIFICATION_CLASS',
                                  'cmpd_classification_alternative_parent_list': 'CMPD_CLASSIFICATION_ALTERNATIVEPARENT'
                              })

        df_meta_unique = df_meta_unique[[
            'CLUSTER_ID', 'GLOBAL_ACCESSION', 'DATASET', 'DS_SPLIT_KEY', 'INCHI', 'INCHI_KEY', 'ACCESSION_NUMBER',
            'CMPD_NAME', 'FILE_NAME', 'PATHWAY_UNIQUE_ID', 'PATHWAY_COMMON_NAME', 'PATHWAY_COMMON_NAMEX',
            'RETENTION_TIME_IN_SEC', 'PRECURSOR_MZ', 'UNIQUE_LEVEL', 'PEAK_LIST_MZ_INT_REL', 'PEAK_LIST_MZ_ANNOID',
            'DIC_ANNOID_STRUCT', 'NUMBER_OF_SPECTRA', 'LIST_ACCESSION_NUMBERS', 'CMPD_CLASSIFICATION_KINGDOM',
            'CMPD_CLASSIFICATION_SUPERCLASS', 'CMPD_CLASSIFICATION_CLASS', 'CMPD_CLASSIFICATION_ALTERNATIVEPARENT'
        ]]

        df_meta_unique.to_csv(path, sep='\t', index=False)


if __name__ == '__main__':
    write_cluster_attribute('../test_cluster_attribute.tsv', 'cmpd_classification_superclass')