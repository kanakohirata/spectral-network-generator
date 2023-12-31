from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import os
import pandas as pd
import re
import shutil


LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


def get_metadata_dtype():
    dtype = [
        ('index', 'u8'),
        ('tag', 'O'),  # str
        ('keyword', 'O'),  # str
        ('cluster_id', 'O'),  # str
        ('cluster_name', 'O'),  # str
        ('source_filename', 'O'),  # str
        ('global_accession', 'O'),  # str
        ('accession_number', 'O'),  # str
        ('precursor_mz', 'f8'),
        ('rt_in_sec', 'f8'),
        ('retention_index', 'f8'),
        ('inchi', 'O'),  # str
        ('inchikey', 'O'),  # str
        ('author', 'O'),  # str
        ('compound_name', 'O'),  # str
        ('title', 'O'),  # str
        ('instrument_type', 'O'),  # str
        ('ionization_mode', 'O'),  # str
        ('fragmentation_type', 'O'),  # str
        ('ion_mode', 'O'),  # str
        ('precursor_type', 'O'),  # str
        ('number_of_peaks', 'u8'),
        ('peaks', 'O'),  # list
        ('mz_list', 'O'),  # list
        ('external_compound_unique_id_list', 'O'),  # list
        ('pathway_unique_id_list', 'O'),  # list
        ('pathway_common_name_list', 'O'),  # list
        ('cmpd_classification_superclass', 'O'),  # str
        ('cmpd_classification_class', 'O'),  # str
        ('cmpd_classification_subclass', 'O'),  # str
        ('cmpd_classification_alternative_parent_list', 'O')  # list
    ]

    return dtype


def initialize_spectrum_metadata_file(dir_path='./spectrum_metadata'):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    else:
        for f in os.listdir(dir_path):
            p = os.path.join(dir_path, f)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.splitext(f)[1] == '.npy':
                os.remove(p)

    dir_path_raw = os.path.join(dir_path, 'raw')
    if not os.path.isdir(dir_path_raw):
        os.makedirs(dir_path_raw)

    dir_path_filtered = os.path.join(dir_path, 'filtered')
    if not os.path.isdir(dir_path_filtered):
        os.makedirs(dir_path_filtered)

    dir_path_grouped = os.path.join(dir_path, 'grouped')
    if not os.path.isdir(dir_path_grouped):
        os.makedirs(dir_path_grouped)


def write_metadata(metadata_path, spectra, export_tsv=False):
    LOGGER.info(f'Write metadata to {os.path.basename(metadata_path)}')

    metadata_dir = os.path.dirname(metadata_path)
    if not os.path.isdir(metadata_dir):
        os.makedirs(metadata_dir)

    metadata_list = []
    dtypes = get_metadata_dtype()
    
    for spectrum in spectra:
        metadata = []
        for dtype in dtypes:
            if dtype[0] == 'index':
                metadata.append(spectrum.get(dtype[0]))
            
            elif dtype[0] in ('tag', 'source_filename', 'global_accession', 'accession_number',
                            'inchi', 'inchikey', 'author', 'title', 'instrument_type'):
                metadata.append(spectrum.get(dtype[0]) or '')

            elif dtype[0] in ('keyword', 'cluster_id', 'cluster_name'):
                metadata.append('')

            elif dtype[0] in ('precursor_mz', 'rt_in_sec', 'retention_index'):
                metadata.append(spectrum.get(dtype[0]) or 0)

            elif dtype[0] == 'compound_name':
                metadata.append(spectrum.get('compound_name') or spectrum.get('name') or '')

            elif dtype[0] == 'ionization_mode':
                ionization_mode = spectrum.get('ionization_mode')\
                                  or spectrum.get('ionizationmode')\
                                  or spectrum.get('ionization')\
                                  or ''

                if not ionization_mode:
                    instrument_type = spectrum.get('instrument_type') or ''
                    matches = re.findall(f'[A-Za-z]+', instrument_type)
                    if 'ESI' in matches:
                        ionization_mode = 'ESI'
                    elif 'EI' in matches:
                        ionization_mode = 'EI'
                    elif 'CI' in matches:
                        ionization_mode = 'CI'
                    elif 'MALDI' in matches:
                        ionization_mode = 'MALDI'
                    elif 'GC' in matches:
                        ionization_mode = 'GC'
                    elif 'APCI' in matches:
                        ionization_mode = 'APCI'

                metadata.append(ionization_mode)

            elif dtype[0] == 'fragmentation_type':
                metadata.append(spectrum.get('fragmentation_type')
                                or spectrum.get('fragmentation_mode')
                                or spectrum.get('fragmentation')
                                or '')

            elif dtype[0] == 'ion_mode':
                metadata.append(spectrum.get('ionmode') or '')

            elif dtype[0] == 'precursor_type':
                metadata.append(spectrum.get('precursor_type')
                                or spectrum.get('precursortype')
                                or spectrum.get('adduct')
                                or '')

            elif dtype[0] == 'number_of_peaks':
                metadata.append(spectrum.mz.size)

            elif dtype[0] == 'peaks':
                metadata.append(spectrum.peaks.to_numpy)

            elif dtype[0] == 'mz_list':
                metadata.append(spectrum.mz)

            elif dtype[0] in ('external_compound_unique_id_list', 'pathway_unique_id_list', 'pathway_common_name_list'):
                metadata.append([])

            elif dtype[0] == 'cmpd_classification_superclass':
                metadata.append(spectrum.get('classification_superclass') or 'noclassification')

            elif dtype[0] == 'cmpd_classification_class':
                metadata.append(spectrum.get('classification_class') or 'noclassification')

            elif dtype[0] == 'cmpd_classification_subclass':
                metadata.append(spectrum.get('classification_subclass') or 'noclassification')

            elif dtype[0] == 'cmpd_classification_alternative_parent_list':
                metadata.append(spectrum.get('classification_alternative_parent') or ['noclassification'])

        metadata_list.append(tuple(metadata))

    metadata_arr = np.array(metadata_list, dtype=dtypes)

    # Append metadata_arr to an already existing array.
    if os.path.isfile(metadata_path):
        existing_metadata_arr = np.load(metadata_path, allow_pickle=True)
        metadata_arr = np.hstack((existing_metadata_arr, metadata_arr))

    with open(metadata_path, 'wb') as f:
        np.save(f, metadata_arr)
        f.flush()

    if export_tsv:
        tsv_path = os.path.splitext(metadata_path)[0] + '.tsv'
        df = pd.DataFrame.from_records(metadata_arr)
        df.to_csv(tsv_path, sep='\t', index=False)


def get_npy_metadata_paths(dir_path) -> list:
    """
    Parameters
    ----------
    dir_path : str

    Returns
    -------
    list
        If dir_path includes 'ref_dataset_0.npy', 'ref_dataset_1.npy', 'ref_dataset_2.npy',
        the following list will be returned.

        list[('dir_path/ref_dataset_0.npy', 0),
             ('dir_path/ref_dataset_1.npy', 1),
             ('dir_path/ref_dataset_2.npy', 2)]
    """
    path_vs_index_list = []

    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and filename.endswith('.npy'):
            matchms = re.findall(r'\d+', filename)
            index = int(matchms[-1])
            path_vs_index_list.append((path, index))

    path_vs_index_list.sort(key=lambda x: x[1])

    return path_vs_index_list


def concatenate_npy_metadata_files(paths, output_path, export_tsv=False):
    if os.path.isfile(output_path):
        os.remove(output_path)

    # Make output folder if it does not exist.
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    arr = None
    for path in paths:
        # Load metadata array.
        arr = np.load(path, allow_pickle=True)

        # Append arr to an already existing array.
        if os.path.isfile(output_path):
            existing_arr = np.load(output_path, allow_pickle=True)
            arr = np.hstack((existing_arr, arr))

        with open(output_path, 'wb') as f:
            np.save(f, arr)
            f.flush()

    if export_tsv and arr is not None:
        tsv_path = os.path.splitext(output_path)[0] + '.tsv'
        df = pd.DataFrame.from_records(arr)
        df.to_csv(tsv_path, sep='\t', index=False)
