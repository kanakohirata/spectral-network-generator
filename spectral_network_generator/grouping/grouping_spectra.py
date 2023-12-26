from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import os
import pickle
from . import grouping_metadata
from my_parser.matchms_spectrum_parser import get_serialized_spectra_paths
from my_parser.spectrum_metadata_parser import get_npy_metadata_paths

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


def group_spectra(spectra_dir, metadata_dir, output_parent_dir):
    LOGGER.info(f'Group and serialized spectra. ({spectra_dir})')
    length_to_export = 1000

    spectra_path_vs_index_list = get_serialized_spectra_paths(spectra_dir)
    metadata_path_vs_index_list = get_npy_metadata_paths(metadata_dir)

    for spectra_path, _, _ in spectra_path_vs_index_list:
        # Load spectra
        with open(spectra_path, 'rb') as f:
            spectra = pickle.load(f)

        # Load metadata
        for metadata_path, metadata_idx in metadata_path_vs_index_list:
            metadata_arr = np.load(metadata_path, allow_pickle=True)
            spectra_index_arr = metadata_arr['index']

            # Get dataset label
            dataset_label = os.path.splitext(os.path.basename(metadata_path))[0]
            
            output_folder_name = dataset_label
            output_dir = os.path.join(output_parent_dir, output_folder_name)
            # Make the output folder if it does not exist.
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            dataset_spectra = []
            start_index = 0

            # If there is already grouped spectra in output_dir, the spectra will be added.
            already_grouped_spectra_path_vs_index_list = get_serialized_spectra_paths(output_dir)
            if already_grouped_spectra_path_vs_index_list:
                last_spectrum_idx = already_grouped_spectra_path_vs_index_list[-1][2]
                if (last_spectrum_idx + 1) % length_to_export != 0:
                    start_index = already_grouped_spectra_path_vs_index_list[-1][1]
                    last_spectra_path = already_grouped_spectra_path_vs_index_list[-1][0]

                    # Load already existing spectra
                    with open(last_spectra_path, 'rb') as f:
                        dataset_spectra = pickle.load(f)

                    os.remove(last_spectra_path)

            # Get dataset spectra
            for spectrum in spectra:
                spectrum_index = spectrum.get('index')
    
                if spectrum_index in spectra_index_arr:
                    dataset_spectra.append(spectrum)
            
                    if len(dataset_spectra) % length_to_export == 0:
                        output_filename = f'{start_index}-{start_index + length_to_export - 1}.pickle'
                        output_path = os.path.join(output_dir, output_filename)

                        with open(output_path, 'wb') as f:
                            pickle.dump(dataset_spectra, f)
                            f.flush()

                        dataset_spectra = []
                        start_index += length_to_export

            if dataset_spectra:
                output_filename = f'{start_index}-{start_index + len(dataset_spectra) - 1}.pickle'
                output_path = os.path.join(output_dir, output_filename)

                with open(output_path, 'wb') as f:
                    pickle.dump(dataset_spectra, f)
                    f.flush()
