import os.path
import pickle
import shutil
from my_parser.matchms_spectrum_parser import get_serialized_spectra_paths


def assign_spectrum_index(spectra_dir, global_index_start):
    spectra_dir_name = os.path.basename(spectra_dir)
    # Get spectra file paths
    spectra_path_and_index_list = get_serialized_spectra_paths(spectra_dir)
    global_index = global_index_start
    source_filename = ''
    for path, _, _ in spectra_path_and_index_list:
        # Load spectra
        with open(path, 'rb') as f:
            spectra = pickle.load(f)

        for spectrum in spectra:
            # Get source filename
            if not source_filename:
                source_filename = spectrum.get('source_filename') or ''
            spectrum.set('index', global_index)
            global_index += 1

        # Output spectra reassigned indexes.
        with open(path, 'wb') as f:
            pickle.dump(spectra, f)
            f.flush()

    global_index -= 1
    return source_filename, spectra_dir_name, global_index
