import os.path

import numpy as np
import pandas as pd


def assign_cluster_id(score_paths, metadata_path):
    # Load metadata array
    metadata_arr = np.load(metadata_path, allow_pickle=True)
    metadata_df = pd.DataFrame.from_records(metadata_arr[['keyword', 'index', 'cluster_id', 'cluster_name', 'global_accession']])

    metadata_df.rename(inplace=True,
                       columns={'keyword': 'keyword_b',
                                'index': 'index_b',
                                'cluster_id': 'cluster_id_b',
                                'cluster_name': 'cluster_name_b',
                                'global_accession': 'global_accession_b'})
    assigned_score_paths = []
    for score_path in score_paths:
        # Load score array
        score_arr = np.load(score_path, allow_pickle=True)
        score_df = pd.DataFrame.from_records(
            score_arr[['index', 'keyword_a', 'keyword_b', 'global_accession_a', 'global_accession_b',
                       'score', 'matches', 'matched_peak_idx_a', 'matched_peak_idx_b']])

        # Merge score_df and metadata_df on 'keyword_a' and 'global_accession_a'
        metadata_df.rename(inplace=True,
                           columns={'keyword_b': 'keyword_a',
                                    'index_b': 'index_a',
                                    'cluster_id_b': 'cluster_id_a',
                                    'cluster_name_b': 'cluster_name_a',
                                    'global_accession_b': 'global_accession_a'})
        score_df = pd.merge(score_df, metadata_df, on=['keyword_a', 'global_accession_a'])

        # Merge score_df and metadata_df on 'keyword_b' and 'global_accession_b'
        metadata_df.rename(inplace=True,
                           columns={'keyword_a': 'keyword_b',
                                    'index_a': 'index_b',
                                    'cluster_id_a': 'cluster_id_b',
                                    'cluster_name_a': 'cluster_name_b',
                                    'global_accession_a': 'global_accession_b'})
        score_df = pd.merge(score_df, metadata_df, on=['keyword_b', 'global_accession_b'])

        # Order columns
        score_df = score_df[list(score_arr.dtype.names)]

        # Remove rows without 'index_a' or 'index_b'
        score_df.dropna(subset=['index_a', 'index_b'], how='any', inplace=True)

        # Update score_arr
        score_arr = np.array(score_df.to_records(index=False), dtype=score_arr.dtype)

        output_dir = os.path.dirname(score_path)
        score_filename = os.path.basename(score_path)
        output_path = os.path.join(output_dir, f'__{score_filename}')

        with open(output_path, 'wb') as f:
            np.save(f, score_arr)
            f.flush()

        assigned_score_paths.append(output_path)

    return assigned_score_paths



