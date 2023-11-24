import h5py
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import pandas as pd
import re
from my_parser.score_parser import get_chunks

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


def write_edge_info(path, score_threshold, minimum_peak_match_to_output, mz_tolerance):
    with h5py.File('./spectrum_metadata.h5', 'r') as h5_meta:
        dset_meta = h5_meta['filtered/metadata']
        df_meta = pd.DataFrame.from_records(dset_meta[()][[
            'global_accession', 'accession_number', 'source_filename', 'precursor_mz', 'rt_in_sec', 'title', 'inchi',
            'peaks', 'mz_list', 'external_compound_unique_id_list', 'pathway_unique_id_list', 'pathway_common_name_list',
            'cmpd_classification_superclass_list', 'cmpd_classification_class_list'
        ]])

        # Change types
        df_meta['accession_number'] = df_meta['accession_number'].str.decode('utf-8')
        df_meta['source_filename'] = df_meta['source_filename'].str.decode('utf-8')
        df_meta['title'] = df_meta['title'].str.decode('utf-8')
        df_meta['inchi'] = df_meta['inchi'].str.decode('utf-8')
        df_meta['peaks'] = df_meta['peaks'].str.decode('utf-8')
        df_meta['mz_list'] = df_meta['mz_list'].str.decode('utf-8')
        df_meta['pathway_unique_id_list'] = df_meta['pathway_unique_id_list'].str.decode('utf-8')
        df_meta['pathway_common_name_list'] = df_meta['pathway_common_name_list'].str.decode('utf-8')
        df_meta['cmpd_classification_superclass_list'] = df_meta['cmpd_classification_superclass_list'].str.decode('utf-8')
        df_meta['cmpd_classification_class_list'] = df_meta['cmpd_classification_class_list'].str.decode('utf-8')

        LOGGER.debug(0)
        # Convert peaks from str to list
        df_meta['peaks'] = df_meta['peaks'].apply(lambda x: re.findall(r'\d+\.\d+', x))
        df_meta['peaks'] = df_meta['peaks'].apply(lambda x: [float(_x) for _x in x])
        df_meta['peaks'] = df_meta['peaks'].apply(lambda x: [list(x[i:i + 2]) for i in range(0, len(x), 2)])
        df_meta['peaks'] = df_meta['peaks'].apply(np.array)
        
        # Convert mz_list from str to np.array
        df_meta['mz_list'] = df_meta['mz_list'].apply(lambda x: re.findall(r'\d+\.\d+', x))
        df_meta['mz_list'] = df_meta['mz_list'].apply(lambda x: [float(_x) for _x in x])
        df_meta['mz_list'] = df_meta['mz_list'].apply(np.array)
        
        # Convert pathway unique id from str to np.array
        df_meta['pathway_unique_id_list'] = df_meta['pathway_unique_id_list'].apply(
            lambda x: re.findall(r'[A-Za-z0-9-]+', x))
        df_meta['pathway_unique_id_list'] = df_meta['pathway_unique_id_list'].apply(np.array)

        # Convert pathway common name from str to np.array
        df_meta['pathway_common_name_list'] = df_meta['pathway_common_name_list'].apply(
            lambda x: x.split(', ') if x else [])
        df_meta['pathway_common_name_list'] = df_meta['pathway_common_name_list'].apply(np.array)

        # Convert compound classes from str to list
        df_meta['cmpd_classification_superclass_list'] =\
            df_meta['cmpd_classification_superclass_list'].apply(lambda x: x.split('|') if x else [])
        df_meta['cmpd_classification_class_list'] =\
            df_meta['cmpd_classification_class_list'].apply(lambda x: x.split('|') if x else [])

        # Add 'cmpd_classification_kingdom_list' column
        df_meta['cmpd_classification_kingdom_list'] =\
            df_meta['cmpd_classification_superclass_list'].apply(
                lambda sl: ['Inorganic compounds' if s in ('Homogeneous non-metal compounds', 'Mixed metal/non-metal compounds') else 'Organic compounds' for s in sl])
        LOGGER.debug(1)

        df_meta.rename(columns={'global_accession': 'global_accession_b',
                                'accession_number': 'accession_number_b',
                                'source_filename': 'source_filename_b',
                                'precursor_mz': 'precursor_mz_b',
                                'rt_in_sec': 'rt_in_sec_b',
                                'title': 'title_b',
                                'inchi': 'inchi_b',
                                'peaks': 'peaks_b',
                                'mz_list': 'mz_list_b',
                                'external_compound_unique_id_list': 'external_compound_unique_id_list_b',
                                'pathway_unique_id_list': 'pathway_unique_id_list_b',
                                'pathway_common_name_list': 'pathway_common_name_list_b',
                                'cmpd_classification_kingdom_list': 'cmpd_classification_kingdom_list_b',
                                'cmpd_classification_superclass_list': 'cmpd_classification_superclass_list_b',
                                'cmpd_classification_class_list': 'cmpd_classification_class_list_b'},
                       inplace=True)

        for arr, chunk_start, chunk_end in get_chunks('clustered_score', db_chunk_size=10000, path='./score.h5'):
            LOGGER.debug(f'Write edge info: {chunk_start} - {chunk_end}')
            arr = arr[(arr['score'] >= score_threshold) & (arr['matches'] >= minimum_peak_match_to_output)]

            if not arr.size:
                continue

            df = pd.DataFrame.from_records(arr)
            df_meta.rename(columns={'global_accession_b': 'global_accession_a',
                                    'accession_number_b': 'accession_number_a',
                                    'source_filename_b': 'source_filename_a',
                                    'precursor_mz_b': 'precursor_mz_a',
                                    'rt_in_sec_b': 'rt_in_sec_a',
                                    'title_b': 'title_a',
                                    'inchi_b': 'inchi_a',
                                    'peaks_b': 'peaks_a',
                                    'mz_list_b': 'mz_list_a',
                                    'external_compound_unique_id_list_b': 'external_compound_unique_id_list_a',
                                    'pathway_unique_id_list_b': 'pathway_unique_id_list_a',
                                    'pathway_common_name_list_b': 'pathway_common_name_list_a',
                                    'cmpd_classification_kingdom_list_b': 'cmpd_classification_kingdom_list_a',
                                    'cmpd_classification_superclass_list_b': 'cmpd_classification_superclass_list_a',
                                    'cmpd_classification_class_list_b': 'cmpd_classification_class_list_a'},
                           inplace=True)
            df = pd.merge(df, df_meta, on='global_accession_a', how='left')
            df_meta.rename(columns={'global_accession_a': 'global_accession_b',
                                    'accession_number_a': 'accession_number_b',
                                    'source_filename_a': 'source_filename_b',
                                    'precursor_mz_a': 'precursor_mz_b',
                                    'rt_in_sec_a': 'rt_in_sec_b',
                                    'title_a': 'title_b',
                                    'inchi_a': 'inchi_b',
                                    'peaks_a': 'peaks_b',
                                    'mz_list_a': 'mz_list_b',
                                    'external_compound_unique_id_list_a': 'external_compound_unique_id_list_b',
                                    'pathway_unique_id_list_a': 'pathway_unique_id_list_b',
                                    'pathway_common_name_list_a': 'pathway_common_name_list_b',
                                    'cmpd_classification_kingdom_list_a': 'cmpd_classification_kingdom_list_b',
                                    'cmpd_classification_superclass_list_a': 'cmpd_classification_superclass_list_b',
                                    'cmpd_classification_class_list_a': 'cmpd_classification_class_list_b'},
                           inplace=True)
            LOGGER.debug(2)
            df = pd.merge(df, df_meta, on='global_accession_b', how='left')

            df['delta_mz'] = abs(df['precursor_mz_a'] - df['precursor_mz_b'])

            df['mz_diff'] = df['mz_list_a'].apply(lambda x: x[:, np.newaxis])
            df['mz_diff'] = df['mz_diff'] - df['mz_list_b']
            df['mz_diff'] = df['mz_diff'].apply(lambda x: np.abs(x) <= mz_tolerance)
            df['matched_mz_index_a'] = df['mz_diff'].apply(lambda x: np.where(x)[0])
            df['matched_mz_index_b'] = df['mz_diff'].apply(lambda x: np.where(x)[1])

            df['matched_mz_a'] = df[['mz_list_a', 'matched_mz_index_a']].apply(
                lambda row: row['mz_list_a'][row['matched_mz_index_a']], axis=1)
            df['matched_mz_b'] = df[['mz_list_b', 'matched_mz_index_b']].apply(
                lambda row: row['mz_list_b'][row['matched_mz_index_b']], axis=1)

            df['matched_peaks_a'] = df[['peaks_a', 'matched_mz_index_a']].apply(
                lambda row: row['peaks_a'][row['matched_mz_index_a']], axis=1)
            df['matched_peaks_b'] = df[['peaks_b', 'matched_mz_index_b']].apply(
                lambda row: row['peaks_b'][row['matched_mz_index_b']], axis=1)
            LOGGER.debug(3)
            # Pathway unique id
            df['y_or_n_pathway_unique_id_matched'] \
                = df[['pathway_unique_id_list_a', 'pathway_unique_id_list_b']].apply(
                lambda row: np.intersect1d(row['pathway_unique_id_list_a'],
                                           row['pathway_unique_id_list_b']),
                axis=1)
            df['y_or_n_pathway_unique_id_matched'] = df['y_or_n_pathway_unique_id_matched'].apply(
                lambda x: 'y' if x.size > 0 else 'n')
            LOGGER.debug(4)
            # Kingdom
            df['y_or_n_cmpd_classification_kingdom_matched'] \
                = df[['cmpd_classification_kingdom_list_a', 'cmpd_classification_kingdom_list_b']].apply(
                lambda row: np.intersect1d(row['cmpd_classification_kingdom_list_a'],
                                           row['cmpd_classification_kingdom_list_b']),
                axis=1)
            df['y_or_n_cmpd_classification_kingdom_matched'] = \
                df['y_or_n_cmpd_classification_kingdom_matched'].apply(lambda x: 'y' if x.size > 0 else 'n')

            # Superclass
            df['y_or_n_cmpd_classification_superclass_matched'] \
                = df[['cmpd_classification_superclass_list_a', 'cmpd_classification_superclass_list_b']].apply(
                lambda row: np.intersect1d(row['cmpd_classification_superclass_list_a'],
                                           row['cmpd_classification_superclass_list_b']),
                axis=1)
            df['y_or_n_cmpd_classification_superclass_matched'] =\
                df['y_or_n_cmpd_classification_superclass_matched'].apply(lambda x: 'y' if x.size > 0 else 'n')

            # Class
            df['y_or_n_cmpd_classification_class_matched'] \
                = df[['cmpd_classification_class_list_a', 'cmpd_classification_class_list_b']].apply(
                lambda row: np.intersect1d(row['cmpd_classification_class_list_a'],
                                           row['cmpd_classification_class_list_b']),
                axis=1)
            df['y_or_n_cmpd_classification_class_matched'] = \
                df['y_or_n_cmpd_classification_class_matched'].apply(lambda x: 'y' if x.size > 0 else 'n')

            df['X_LEN_MATCHED_MZ'] = df['matches']
            df['Y_LEN_MATCHED_MZ'] = df['matches']

            df = df[[
                'cluster_id_a', 'cluster_id_b', 'accession_number_a', 'accession_number_b',
                'global_accession_a', 'global_accession_b', 'source_filename_a', 'source_filename_b', 'score', 'matches',
                'delta_mz', 'precursor_mz_a', 'precursor_mz_b', 'rt_in_sec_a', 'rt_in_sec_b', 'title_a', 'title_b',
                'inchi_a', 'inchi_b', 'inchikey_a', 'inchikey_b', 'matched_mz_a', 'matched_mz_b',
                'matched_peaks_a', 'matched_peaks_b',
                'X_LEN_MATCHED_MZ', 'Y_LEN_MATCHED_MZ',
                'pathway_unique_id_list_a', 'pathway_unique_id_list_b',
                'pathway_common_name_list_a', 'pathway_common_name_list_b',
                'y_or_n_pathway_unique_id_matched',
                'cmpd_classification_kingdom_list_a', 'cmpd_classification_kingdom_list_b',
                'y_or_n_cmpd_classification_kingdom_matched',
                'cmpd_classification_superclass_list_a', 'cmpd_classification_superclass_list_b',
                'y_or_n_cmpd_classification_superclass_matched',
                'cmpd_classification_class_list_a', 'cmpd_classification_class_list_b',
                'y_or_n_cmpd_classification_class_matched'
            ]]
            LOGGER.debug(5)

            # Change types
            df['cluster_id_a'] = df['cluster_id_a'].str.decode('utf-8')
            df['cluster_id_b'] = df['cluster_id_b'].str.decode('utf-8')
            df['global_accession_a'] = df['global_accession_a'].str.decode('utf-8')
            df['global_accession_b'] = df['global_accession_b'].str.decode('utf-8')
            df['inchikey_a'] = df['inchikey_a'].str.decode('utf-8')
            df['inchikey_b'] = df['inchikey_b'].str.decode('utf-8')
            df['matched_mz_a'] = df['matched_mz_a'].apply(lambda _a: _a.tolist())
            df['matched_mz_b'] = df['matched_mz_b'].apply(lambda _a: _a.tolist())
            df['matched_peaks_a'] = df['matched_peaks_a'].apply(lambda _a: _a.tolist())
            df['matched_peaks_b'] = df['matched_peaks_b'].apply(lambda _a: _a.tolist())
            df['pathway_unique_id_list_a'] = df['pathway_unique_id_list_a'].apply(lambda _a: _a.tolist())
            df['pathway_unique_id_list_b'] = df['pathway_unique_id_list_b'].apply(lambda _a: _a.tolist())
            df['pathway_common_name_list_a'] = df['pathway_common_name_list_a'].apply(lambda _a: _a.tolist())
            df['pathway_common_name_list_b'] = df['pathway_common_name_list_b'].apply(lambda _a: _a.tolist())
            LOGGER.debug(6)
            df.rename(columns={
                'cluster_id_a': 'X_CLUSTERID',
                'cluster_id_b': 'Y_CLUSTERID',
                'accession_number_a': 'REP_SPECTRUM_X_ACCESSION',
                'accession_number_b': 'REP_SPECTRUM_Y_ACCESSION',
                'global_accession_a': 'REP_SPECTRUM_X_GLOBAL_ACCESSION',
                'global_accession_b': 'REP_SPECTRUM_Y_GLOBAL_ACCESSION',
                'source_filename_a': 'SPECTRUM_X_FILENAME',
                'source_filename_b': 'SPECTRUM_Y_FILENAME',
                'score': 'SCORE',
                'matches': 'NUM_MATCHED_mz_list',
                'delta_mz': 'DELTA_MZ',
                'precursor_mz_a': 'X_PRECURSOR_MZ',
                'precursor_mz_b': 'Y_PRECURSOR_MZ',
                'rt_in_sec_a': 'X_PRECURSOR_RT',
                'rt_in_sec_b': 'Y_PRECURSOR_RT',
                'title_a': 'X_TITLE',
                'title_b': 'Y_TITLE',
                'inchi_a': 'X_INCHI',
                'inchi_b': 'Y_INCHI',
                'inchikey_a': 'X_INCHI_KEY',
                'inchikey_b': 'Y_INCHI_KEY',
                'matched_mz_a': 'X_MATCHED_MZ',
                'matched_mz_b': 'Y_MATCHED_MZ',
                'matched_peaks_a': 'X_MATCHED_PEAKS',
                'matched_peaks_b': 'Y_MATCHED_PEAKS',
                'pathway_unique_id_list_a': '1',
                'pathway_unique_id_list_b': '2',
                'pathway_common_name_list_a': '3',
                'pathway_common_name_list_b': '4',
                'y_or_n_pathway_unique_id_matched': '5',
                'cmpd_classification_kingdom_list_a': '6',
                'cmpd_classification_kingdom_list_b': '7',
                'y_or_n_cmpd_classification_kingdom_matched': '8',
                'cmpd_classification_superclass_list_a': '9',
                'cmpd_classification_superclass_list_b': '10',
                'y_or_n_cmpd_classification_superclass_matched': '11',
                'cmpd_classification_class_list_a': '12',
                'cmpd_classification_class_list_b': '13',
                'y_or_n_cmpd_classification_class_matched': '14'
            }, inplace=True)

            LOGGER.debug(7)

            # temp_path = path + f'.temp{count}'
            # df.to_csv(temp_path, sep='\t', index=False, mode='w')
            if chunk_start == 0:
                df.to_csv(path, sep='\t', index=False, mode='w')
            else:
                df.to_csv(path, sep='\t', index=False, header=False, mode='a')


if __name__ == '__main__':
    write_edge_info('../test_edge_info.tsv', 0.6, 0.5)
