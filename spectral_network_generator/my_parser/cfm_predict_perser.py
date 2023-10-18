import json
from logging import DEBUG, Formatter, getLogger, StreamHandler
import numpy as np
import os
import re
from filter.filter import filter_reference_spectra, remove_blank_spectra_from_sample_spectra
from my_parser import read_metacyc_pathway_compound as read_meta
from my_parser.cluster_attribute_parser import write_cluster_attribute
from my_parser.edge_info_parser import write_edge_info
from my_parser.matchms_spectrum_parser import (delete_serialize_spectra_file, load_and_serialize_spectra,
                                               serialize_filtered_spectra)
from my_parser.score_parser import initialize_score_hdf5
from my_parser.spectrum_metadata_parser import initialize_spectrum_metadata_hdf5
from score.score import calculate_similarity_score, clustering_based_on_inchikey
from utils import add_classyfire_class, add_metacyc_compound_info
from utils.clustering import add_cluster_id


LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False


def convert_cfm_to_json(dir_path, output_path):
    LOGGER.debug('Convert CFM prediction files to a JSON file.')
    paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    if not paths:
        LOGGER.warning(f'No CFM prediction files in {dir_path}')
        return
    
    spectra = []

    for path in paths:
        json_data = {}
        peaks_energy0 = []
        peaks_energy1 = []
        peaks_energy2 = []
        mz_energy2 = []
        intensity_energy2 = []
        flag_energy = -1
        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('#'):
                    _parse_metadata(line, json_data)

                elif line == 'energy0':
                    flag_energy = 0
                elif line == 'energy1':
                    flag_energy = 1
                elif line == 'energy2':
                    flag_energy = 2

                elif re.match(r'(\d+\.\d+) (\d+\.\d+)', line):
                    mz, intensity = re.match(r'(\d+\.\d+) (\d+\.\d+)', line).groups()[:2]
                    

                    if flag_energy == 0:
                        peaks_energy0.append((mz, intensity))
                    elif flag_energy == 1:
                        peaks_energy1.append((mz, intensity))
                    elif flag_energy == 2:
                        peaks_energy2.append((mz, intensity))
                        mz_energy2.append(mz)
                        intensity = float(intensity)
                        intensity_energy2.append(intensity)

                elif not line:
                    break

            median_intensity = np.median(intensity_energy2)
            for mz, intensity in peaks_energy0:
                if mz not in mz_energy2:
                    peaks_energy2.append((mz, median_intensity))
                    mz_energy2.append(mz)

            for mz, intensity in peaks_energy1:
                if mz not in mz_energy2:
                    peaks_energy2.append((mz, median_intensity))
                    mz_energy2.append(mz)

            if not peaks_energy2:
                continue

            peaks_energy2 = list(map(lambda x: [float(x[0]), x[1]], peaks_energy2))
            peaks_energy2 = sorted(peaks_energy2, key=lambda x: x[0])
            peaks_energy2 = list(map(lambda x: f'[{x[0]},{x[1]}]', peaks_energy2))

            json_data['peaks_json'] = '[' + ','.join(peaks_energy2) + ']'

            if json_data['Adduct'] in ('[M+H]+','[M]+','[M+NH4]+','[M+Na]+','[M+K]+','[M+Li]+'):
                json_data['Ion_Mode'] = 'Positive'
            elif json_data['Adduct'] in ('[M-H]-','[M]-','[M+Cl]-','[M+HCOOH-H]-','[M+CH3COOH-H]-','[M-2H]2-'):
                json_data['Ion_Mode'] = 'Negative'

            spectra.append(json_data)

    with open(output_path, 'w') as j:
        json.dump(spectra, j, indent=4)

    print(1)


def _parse_metadata(line:str, data:dict):
    line = line.lstrip('#')
    if re.match(r'In-silico ESI-MS/MS (\[.+].+) Spectra', line):
        precursor_type = re.match(r'In-silico ESI-MS/MS (\[.+].+) Spectra', line).groups()[0]
        data['Adduct'] = precursor_type

    elif line.startswith('PREDICTED BY CFM-ID'):
        data['in_silico_tool'] = line
    
    elif line.startswith('ID='):
        id_ = line.replace('ID=', '')
        data['spectrum_id'] = id_
        data['accession'] = id_

    elif line.startswith('InChI='):
        data['INCHI'] = line
    
    elif line.startswith('InChiKey='):
        data['InChIKey'] = line.replace('InChiKey=', '')
    

if __name__ == '__main__':
    convert_cfm_to_json('../input/cfm_predict')
