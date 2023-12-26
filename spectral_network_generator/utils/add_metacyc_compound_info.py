from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import os
import pandas as pd

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


def add_metacyc_compound_info(metacyc_compound_path, metadata_path, export_tsv=False):
    LOGGER.debug('Add metacyc compound info to metadata.')

    if not os.path.isfile(metacyc_compound_path):
        LOGGER.warning(f'The file is not found: {metacyc_compound_path}')
        return
    # Load metacyc compound array
    arr_metacyc_compound = np.load(metacyc_compound_path, allow_pickle=True)
    if not arr_metacyc_compound.size:
        LOGGER.warning(f'The metacyc compound file has no record: {metacyc_compound_path}')
        return

    # load metadata array
    arr_metadata = np.load(metadata_path, allow_pickle=True)

    df_metadata = pd.DataFrame.from_records(arr_metadata)
    df_metacyc_compound = pd.DataFrame.from_records(arr_metacyc_compound)

    # Add column for the first part of InChIKey.
    df_metadata['inchikey_head'] = df_metadata['inchikey'].str.split('-', expand=True)[0]
    df_metacyc_compound['inchikey_head'] = df_metacyc_compound['inchikey'].str.split('-', expand=True)[0]

    for _i, row in df_metadata[df_metadata['inchikey_head'] != ''].iterrows():
        mask = df_metacyc_compound['inchikey_head'].apply(lambda x: row['inchikey_head'] == x)

        if np.any(mask):
            external_compound_unique_ids = list(df_metacyc_compound[mask]['unique_id'])

            pathway_unique_ids = list(set(i for i in df_metacyc_compound[mask]['pathway_unique_id_list'] if i))
            pathway_common_names = list(set(n for n in df_metacyc_compound[mask]['pathway_common_name_list'] if n))

            df_metadata.loc[_i, 'external_compound_unique_id_list':'pathway_common_name_list'] = [
                external_compound_unique_ids, pathway_unique_ids, pathway_common_names
            ]

    df_metadata = df_metadata.loc[:, arr_metadata.dtype.names[0]:arr_metadata.dtype.names[-1]]
    new_arr = np.array(
        df_metadata.to_records(index=False),
        dtype=arr_metadata.dtype
    )

    with open(metadata_path, 'wb') as f:
        np.save(f, new_arr)
        f.flush()

    if export_tsv:
        tsv_path = os.path.splitext(metadata_path)[0] + '.tsv'
        df_metadata.to_csv(tsv_path, sep='\t', index=False)
