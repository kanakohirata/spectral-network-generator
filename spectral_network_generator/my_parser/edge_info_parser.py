from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import pandas as pd
from utils import split_array

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


def write_edge_info(output_path, score_paths, metadata_path, score_threshold, minimum_peak_match_to_output):
    """
    Parameters
    ----------
    output_path : str
        Path of edge information file.
    score_paths : str
        Paths of spectral similarity score files.
    metadata_path : str
        Path of metadata file.
    score_threshold : float
        Edges with score >= score_threshold will be exported.
    minimum_peak_match_to_output
        Edges with number of matched peaks >= minimum_peak_match_to_output will be exported.

    Returns
    -------

    """
    metadata_arr = np.load(metadata_path, allow_pickle=True)
    df_meta = pd.DataFrame.from_records(metadata_arr)

    del metadata_arr

    df_meta = df_meta[[
        'index', 'accession_number', 'source_filename',
        'precursor_mz', 'rt_in_sec', 'title', 'inchi', 'inchikey', 'peaks', 'mz_list',
        'external_compound_unique_id_list', 'pathway_unique_id_list', 'pathway_common_name_list',
        'cmpd_classification_superclass', 'cmpd_classification_class'
    ]]

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

    df_meta.rename(columns={'index': 'index_b',
                            'accession_number': 'accession_number_b',
                            'source_filename': 'source_filename_b',
                            'precursor_mz': 'precursor_mz_b',
                            'rt_in_sec': 'rt_in_sec_b',
                            'title': 'title_b',
                            'inchi': 'inchi_b',
                            'inchikey': 'inchikey_b',
                            'peaks': 'peaks_b',
                            'mz_list': 'mz_list_b',
                            'external_compound_unique_id_list': 'external_compound_unique_id_list_b',
                            'pathway_unique_id_list': 'pathway_unique_id_list_b',
                            'pathway_common_name_list': 'pathway_common_name_list_b',
                            'cmpd_classification_kingdom': 'cmpd_classification_kingdom_b',
                            'cmpd_classification_superclass': 'cmpd_classification_superclass_b',
                            'cmpd_classification_class': 'cmpd_classification_class_b'},
                   inplace=True)

    writing_count = 0

    for score_path in score_paths:
        LOGGER.debug(f'Write edge info: {score_path}')
        _arr = np.load(score_path, allow_pickle=True)
        _arr = _arr[(_arr['score'] >= score_threshold) & (_arr['matches'] >= minimum_peak_match_to_output)]

        if not _arr.size:
            continue

        for arr, _, _ in split_array(_arr, 10000):
            # Convert score array to a dataframe, and then merge score and metadata dataframes.
            df = pd.DataFrame.from_records(arr)
            df_meta.rename(columns={'index_b': 'index_a',
                                    'accession_number_b': 'accession_number_a',
                                    'source_filename_b': 'source_filename_a',
                                    'precursor_mz_b': 'precursor_mz_a',
                                    'rt_in_sec_b': 'rt_in_sec_a',
                                    'title_b': 'title_a',
                                    'inchi_b': 'inchi_a',
                                    'inchikey_b': 'inchikey_a',
                                    'peaks_b': 'peaks_a',
                                    'mz_list_b': 'mz_list_a',
                                    'external_compound_unique_id_list_b': 'external_compound_unique_id_list_a',
                                    'pathway_unique_id_list_b': 'pathway_unique_id_list_a',
                                    'pathway_common_name_list_b': 'pathway_common_name_list_a',
                                    'cmpd_classification_kingdom_b': 'cmpd_classification_kingdom_a',
                                    'cmpd_classification_superclass_b': 'cmpd_classification_superclass_a',
                                    'cmpd_classification_class_b': 'cmpd_classification_class_a'},
                           inplace=True)
            df = pd.merge(df, df_meta, on='index_a', how='left')
            df_meta.rename(columns={'index_a': 'index_b',
                                    'accession_number_a': 'accession_number_b',
                                    'source_filename_a': 'source_filename_b',
                                    'precursor_mz_a': 'precursor_mz_b',
                                    'rt_in_sec_a': 'rt_in_sec_b',
                                    'title_a': 'title_b',
                                    'inchi_a': 'inchi_b',
                                    'inchikey_a': 'inchikey_b',
                                    'peaks_a': 'peaks_b',
                                    'mz_list_a': 'mz_list_b',
                                    'external_compound_unique_id_list_a': 'external_compound_unique_id_list_b',
                                    'pathway_unique_id_list_a': 'pathway_unique_id_list_b',
                                    'pathway_common_name_list_a': 'pathway_common_name_list_b',
                                    'cmpd_classification_kingdom_a': 'cmpd_classification_kingdom_b',
                                    'cmpd_classification_superclass_a': 'cmpd_classification_superclass_b',
                                    'cmpd_classification_class_a': 'cmpd_classification_class_b'},
                           inplace=True)

            df = pd.merge(df, df_meta, on='index_b', how='left')

            # Get delta of precursor m/z values
            df['delta_mz'] = df['precursor_mz_a'] - df['precursor_mz_b']

            # Change type of matched_peak_idx from list[float] to list[int]
            df['matched_peak_idx_a'] = df['matched_peak_idx_a'].apply(lambda indexes: list(map(int, indexes)))
            df['matched_peak_idx_b'] = df['matched_peak_idx_b'].apply(lambda indexes: list(map(int, indexes)))

            # Get matched m/z values
            df['matched_mz_a'] = df[['mz_list_a', 'matched_peak_idx_a']].apply(
                lambda row: row['mz_list_a'][row['matched_peak_idx_a']], axis=1)
            df['matched_mz_b'] = df[['mz_list_b', 'matched_peak_idx_b']].apply(
                lambda row: row['mz_list_b'][row['matched_peak_idx_b']], axis=1)

            # Get matched peaks
            df['matched_peaks_a'] = df[['peaks_a', 'matched_peak_idx_a']].apply(
                lambda row: row['peaks_a'][row['matched_peak_idx_a']], axis=1)
            df['matched_peaks_b'] = df[['peaks_b', 'matched_peak_idx_b']].apply(
                lambda row: row['peaks_b'][row['matched_peak_idx_b']], axis=1)

            # Get matched pathway unique id
            df['y_or_n_pathway_unique_id_matched'] \
                = df[['pathway_unique_id_list_a', 'pathway_unique_id_list_b']].apply(
                lambda row: np.intersect1d(row['pathway_unique_id_list_a'],
                                           row['pathway_unique_id_list_b']),
                axis=1)
            df['y_or_n_pathway_unique_id_matched'] = df['y_or_n_pathway_unique_id_matched'].apply(
                lambda x: 'y' if x.size > 0 else 'n')

            # Get whether kingdom values match
            df['y_or_n_cmpd_classification_kingdom_matched'] \
                = df[['cmpd_classification_kingdom_a', 'cmpd_classification_kingdom_b']].apply(
                lambda row: 'n'
                            if (row['cmpd_classification_kingdom_a'] == 'noclassification'
                                and row['cmpd_classification_kingdom_b'] == 'noclassification')
                            else (
                                'y'
                                if row['cmpd_classification_kingdom_a'] == row['cmpd_classification_kingdom_b']
                                else 'n'
                            ),
                axis=1)

            # Get whether superclass values match
            df['y_or_n_cmpd_classification_superclass_matched'] \
                = df[['cmpd_classification_superclass_a', 'cmpd_classification_superclass_b']].apply(
                lambda row: 'n'
                            if (row['cmpd_classification_superclass_a'] == 'noclassification'
                                and row['cmpd_classification_superclass_b'] == 'noclassification')
                            else (
                                'y'
                                if row['cmpd_classification_superclass_a'] == row['cmpd_classification_superclass_b']
                                else 'n'
                            ),
                axis=1)

            # Get whether class values match
            df['y_or_n_cmpd_classification_class_matched'] \
                = df[['cmpd_classification_class_a', 'cmpd_classification_class_b']].apply(
                lambda row: 'n'
                            if (row['cmpd_classification_class_a'] == 'noclassification'
                                and row['cmpd_classification_class_b'] == 'noclassification')
                            else (
                                'y'
                                if row['cmpd_classification_class_a'] == row['cmpd_classification_class_b']
                                else 'n'
                            ),
                axis=1)
            
            # Change type of classification from str to list ----------------------------
            # !!! This changing is temporary for spectral network visualizer.
            #     and will be removed in the future.
            df['cmpd_classification_kingdom_a'] = df['cmpd_classification_kingdom_a'].apply(lambda class_: [class_,])
            df['cmpd_classification_kingdom_b'] = df['cmpd_classification_kingdom_b'].apply(lambda class_: [class_,])
            df['cmpd_classification_superclass_a'] = df['cmpd_classification_superclass_a'].apply(lambda class_: [class_,])
            df['cmpd_classification_superclass_b'] = df['cmpd_classification_superclass_b'].apply(lambda class_: [class_,])
            df['cmpd_classification_class_a'] = df['cmpd_classification_class_a'].apply(lambda class_: [class_,])
            df['cmpd_classification_class_b'] = df['cmpd_classification_class_b'].apply(lambda class_: [class_,])
            # ---------------------------------------------------------------------------

            # Convert type of 'matched_mz' and 'matched_peaks' from ndarray to list
            df['matched_mz_a'] = df['matched_mz_a'].apply(list)
            df['matched_mz_b'] = df['matched_mz_b'].apply(list)
            df['matched_peaks_a'] = df['matched_peaks_a'].apply(lambda peaks: peaks.tolist())
            df['matched_peaks_b'] = df['matched_peaks_b'].apply(lambda peaks: peaks.tolist())

            df['X_LEN_MATCHED_MZ'] = df['matches']
            df['Y_LEN_MATCHED_MZ'] = df['matches']

            df = df[[
                'cluster_id_a', 'cluster_id_b',
                'accession_number_a', 'accession_number_b',
                'global_accession_a', 'global_accession_b',
                'source_filename_a', 'source_filename_b',
                'score', 'matches', 'delta_mz',
                'precursor_mz_a', 'precursor_mz_b',
                'rt_in_sec_a', 'rt_in_sec_b',
                'title_a', 'title_b',
                'inchi_a', 'inchi_b',
                'inchikey_a', 'inchikey_b',
                'matched_mz_a', 'matched_mz_b',
                'matched_peaks_a', 'matched_peaks_b',
                'X_LEN_MATCHED_MZ', 'Y_LEN_MATCHED_MZ',
                'pathway_unique_id_list_a', 'pathway_unique_id_list_b',
                'pathway_common_name_list_a', 'pathway_common_name_list_b',
                'y_or_n_pathway_unique_id_matched',
                'cmpd_classification_kingdom_a', 'cmpd_classification_kingdom_b',
                'y_or_n_cmpd_classification_kingdom_matched',
                'cmpd_classification_superclass_a', 'cmpd_classification_superclass_b',
                'y_or_n_cmpd_classification_superclass_matched',
                'cmpd_classification_class_a', 'cmpd_classification_class_b',
                'y_or_n_cmpd_classification_class_matched'
            ]]

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
                'matches': 'NUM_MATCHED_PEAKS',
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
                'cmpd_classification_kingdom_a': '6',
                'cmpd_classification_kingdom_b': '7',
                'y_or_n_cmpd_classification_kingdom_matched': '8',
                'cmpd_classification_superclass_a': '9',
                'cmpd_classification_superclass_b': '10',
                'y_or_n_cmpd_classification_superclass_matched': '11',
                'cmpd_classification_class_a': '12',
                'cmpd_classification_class_b': '13',
                'y_or_n_cmpd_classification_class_matched': '14'
            }, inplace=True)

            # temp_path = path + f'.temp{count}'
            # df.to_csv(temp_path, sep='\t', index=False, mode='w')
            if writing_count == 0:
                df.to_csv(output_path, sep='\t', index=False, mode='w')
            else:
                df.to_csv(output_path, sep='\t', index=False, header=False, mode='a')

            writing_count += 1

