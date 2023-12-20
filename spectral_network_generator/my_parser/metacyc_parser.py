__version__ = 'a2'
'''
* Version a1 ----------------------
E. Hayakawa
last modified 20170713
read compound fixed
* ---------------------------------

HOW TO USE
# create list of pathway object ===========================

# declare / create empty list
list_pathway_obj = []
""":type: list[MetaCycPathway]"""

# feed the filename to the function--------------
# list_pathway_obj will have list of MetaCycPathway

list_pathway_obj = read_metacyc_pathway_compound_a1.read_midas_pathways_dat( filename )

# you can use the list as you like....
# refer to "MetaCycPathway" class declaration for more details

for o in list_pathway_obj:
    LOGGER.debug " "
    LOGGER.debug "id:", o.unique_id

    LOGGER.debug o.list_compound_unique_id

LOGGER.debug "processing compound list "

# reading metacyc-style compound file =================================
# normally the file name should be like ..........

"""
UNIQUE-ID - D-ALANINE
TYPES - D-Amino-Acids
COMMON-NAME - D-alanine
ABBREV-NAME - D-ala
...
INCHI - inchi=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m1/s1
INCHI-KEY - InChIKey=QNAYBMKLOCPYGJ-UWTATZPHSA-N
...
//

"""
HOW TO USE

# tell filename
filename_cmpd = "test_compounds.dat"

# feed the file name to the function
# list_compound_obj will have a list of "MetaCycCompound"

list_compound_obj = read_metacyc_pathway_compound_a1.read_midas_compounds_dat( filename_cmpd )
'''
import h5py
from logging import DEBUG, Formatter, getLogger, StreamHandler
import networkx as nx
import numpy as np
import numpy.lib.recfunctions as rfn
from rdkit import Chem
import re
import os.path
import pandas as pd

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.propagate = False

H5PY_STR_TYPE = h5py.special_dtype(vlen=str)


class MetaCycPathway:
    def __init__(self):
        self.unique_id = ""
        self.types = ""
        self.common_name = ""
        self.list_compound_unique_id = []
        self.nx_directed_graph = nx.DiGraph


class MetaCycCompound:
    def __init__(self):
        self.unique_id = ""
        self.types = ""
        self.name = ""
        self.common_name = ""
        self.inchi = ""
        self.inchikey = ""
        self.smiles = ""
        self.monoisotopic_mw = 0
        self.ec = ""
        self.reaction_equation = ""
        self.list_pathway_unique_id = []
        self.list_pathway_common_name = []
        self.pathway = ""


def initialize_metacyc_hdf5(path):
    LOGGER.debug(f'Initialize {path}')
    with h5py.File(path, 'w') as h5:
        h5.create_dataset('compound', dtype='f')
        h5.create_dataset('pathway', dtype='f')


def get_metacyc_compound_dtype():
    dtype = [
        ('filename', 'O'),  # str
        ('unique_id', 'O'),  # str
        ('common_name', 'O'),  # str
        ('inchikey', 'O'),  # str
        ('inchi', 'O'),  # str
        ('smiles', 'O'),  # str
        ('pathway_unique_id_list', 'O'),  # list
        ('pathway_common_name_list', 'O')  # list
    ]
    return dtype


def get_metacyc_pathway_dtype():
    dtype = [
        ('filename', 'O'),  # str
        ('unique_id', 'O'),  # str
        ('common_name', 'O'),  # str
        ('compound_unique_id_list', 'O')  # list
    ]
    return dtype


def convert_metacyc_compounds_dat_to_h5(path, output_path, parameters_to_open_file=None):
    LOGGER.debug(f'Read {path}')

    if not isinstance(parameters_to_open_file, dict):
        parameters_to_open_file = dict(encoding='utf8')
    filename = os.path.basename(path)

    if not os.path.isfile(output_path):
        initialize_metacyc_hdf5(output_path)

    # flags
    flag_hit_inchi_key = 0

    data = []
    with open(path, 'r', **parameters_to_open_file) as f:
        for each_line in f:
            if each_line.startswith("//"):
                #  end of a compound info
                # if you did not find inchi key, you have to generate it.
                if flag_hit_inchi_key == 0 and len(inchi) > 2:
                    inchikey = Chem.InchiToInchiKey(inchi)

                # append it,
                data.append((
                    filename, unique_id, common_name, inchikey, inchi, smiles,
                ))

                # initialize flags
                flag_hit_inchi_key = 0

            # get common name of compound ------------------------------
            #  common_name: COMMON-NAME - 2-nitrogenization
            elif each_line.startswith("COMMON-NAME"):
                common_name = str(each_line.split(" - ")[1].strip("\n"))

            # get unique id of compound ------------------------------
            #  unique_id: UNIQUE-ID - PWY-6398
            elif each_line.startswith("UNIQUE-ID"):
                unique_id = str(each_line.split(" - ")[1].strip("\n"))

            # get inchi string ------------------------------
            #  inchi: INCHI - INCHI=1S/C7H15N2O8P/c8-1-4(10)9-7-6(12)5(11)3(17-7)2-16-18(13,14)1
            elif each_line.startswith("INCHI -"):
                inchi = str(each_line.replace('INCHI - ', '')).strip("\n")

            # get inchi key ------------------------------
            #  inchikey: INCHI-KEY - InChIKey=KFWWCMJSYSSPSK-BOGFJHSMSA-J
            elif each_line.startswith("INCHI-KEY"):
                flag_hit_inchi_key = 1
                inchikey = str(each_line.replace('INCHI-KEY - InChIKey=', '')).strip("\n")

            # get smile string ------------------------------
            #  smile: SMILES - CNC(=O)CCC([N+])C(=O)[O-]
            elif each_line.startswith("SMILES - "):
                smiles = str(each_line.replace('SMILES - ', '')).strip("\n")

    with h5py.File(output_path, 'a') as h5:
        dset = h5['compound']
        _arr = np.array(
            data,
            dtype=[('filename', H5PY_STR_TYPE), ('unique_id', H5PY_STR_TYPE), ('common_name', H5PY_STR_TYPE),
                   ('inchikey', H5PY_STR_TYPE), ('inchi', H5PY_STR_TYPE), ('smiles', H5PY_STR_TYPE)]
        )

        if not dset.shape:
            del h5['compound']
            h5.create_dataset('compound', data=_arr, shape=_arr.shape[0], maxshape=(None,))
        else:
            current_length = dset.len()
            dset.resize((current_length + _arr.shape[0]), axis=0)
            dset[current_length:] = _arr

        h5.flush()


def convert_metacyc_compounds_dat_to_npy(path, output_path, parameters_to_open_file=None):
    LOGGER.debug(f'Read {path}')

    if not isinstance(parameters_to_open_file, dict):
        parameters_to_open_file = dict(encoding='utf8')
    filename = os.path.basename(path)

    # Make output folder if it does not exist.
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # flags
    flag_hit_inchi_key = 0

    dtypes = get_metacyc_compound_dtype()

    data = []
    with open(path, 'r', **parameters_to_open_file) as f:
        for each_line in f:
            if each_line.startswith("//"):
                #  end of a compound info
                # if you did not find inchi key, you have to generate it.
                if flag_hit_inchi_key == 0 and len(inchi) > 2:
                    inchikey = Chem.InchiToInchiKey(inchi)

                # append it,
                current_data = []
                for field_name, _ in dtypes:
                    if field_name == 'filename':
                        current_data.append(filename)
                    elif field_name == 'unique_id':
                        current_data.append(unique_id)
                    elif field_name == 'common_name':
                        current_data.append(common_name)
                    elif field_name == 'inchikey':
                        current_data.append(inchikey)
                    elif field_name == 'inchi':
                        current_data.append(inchi)
                    elif field_name == 'smiles':
                        current_data.append(smiles)
                    elif field_name == 'pathway_unique_id_list':
                        current_data.append([])
                    elif field_name == 'pathway_common_name_list':
                        current_data.append([])

                data.append(tuple(current_data))

                # initialize flags
                flag_hit_inchi_key = 0

            # get common name of compound ------------------------------
            #  common_name: COMMON-NAME - 2-nitrogenization
            elif each_line.startswith("COMMON-NAME"):
                common_name = str(each_line.split(" - ")[1].strip("\n"))

            # get unique id of compound ------------------------------
            #  unique_id: UNIQUE-ID - PWY-6398
            elif each_line.startswith("UNIQUE-ID"):
                unique_id = str(each_line.split(" - ")[1].strip("\n"))

            # get inchi string ------------------------------
            #  inchi: INCHI - INCHI=1S/C7H15N2O8P/c8-1-4(10)9-7-6(12)5(11)3(17-7)2-16-18(13,14)1
            elif each_line.startswith("INCHI -"):
                inchi = str(each_line.replace('INCHI - ', '')).strip("\n")

            # get inchi key ------------------------------
            #  inchikey: INCHI-KEY - InChIKey=KFWWCMJSYSSPSK-BOGFJHSMSA-J
            elif each_line.startswith("INCHI-KEY"):
                flag_hit_inchi_key = 1
                inchikey = str(each_line.replace('INCHI-KEY - InChIKey=', '')).strip("\n")

            # get smile string ------------------------------
            #  smile: SMILES - CNC(=O)CCC([N+])C(=O)[O-]
            elif each_line.startswith("SMILES - "):
                smiles = str(each_line.replace('SMILES - ', '')).strip("\n")

    arr = np.array(data, dtype=dtypes)

    # Add already existing array to arr.
    if os.path.isfile(output_path):
        existing_arr = np.load(output_path, allow_pickle=True)
        arr = np.hstack((existing_arr, arr))

    np.save(output_path, arr)


def read_metacyc_compounds_dat(filepath, output_path, parameters_to_open_file=None):
    if not isinstance(parameters_to_open_file, dict):
        parameters_to_open_file = dict(encoding='utf8')
    filename = os.path.basename(filepath)

    list_compound_obj = []  # type: list[MetaCycCompound]
    # need to create first object
    curr_compound_obj = MetaCycCompound()
    # flags
    flag_hit_inchi_key = 0

    data = []
    with open(filepath, 'r', **parameters_to_open_file) as f:
        for each_line in f:
            # first line
            if each_line.startswith("//"):
                #  end of a compound info
                # if you did not find inchi key, you have to generate it.
                if flag_hit_inchi_key == 0 and len(curr_compound_obj.inchi) > 2:
                    curr_compound_obj.inchikey = Chem.InchiToInchiKey(curr_compound_obj.inchi)

                # append it,
                list_compound_obj.append(curr_compound_obj)
                data.append((
                    filename,
                    curr_compound_obj.unique_id,
                    curr_compound_obj.common_name,
                    curr_compound_obj.inchikey,
                    curr_compound_obj.inchi,
                    curr_compound_obj.smiles,
                ))

                # then create new
                curr_compound_obj = MetaCycCompound()
                # initialize flags
                flag_hit_inchi_key = 0

            # get common name of compound ------------------------------
            #  common_name: COMMON-NAME - 2-nitrogenization
            elif each_line.startswith("COMMON-NAME"):
                curr_compound_obj.common_name = str(each_line.split(" - ")[1].strip("\n"))

            # get unique id of compound ------------------------------
            #  unique_id: UNIQUE-ID - PWY-6398
            elif each_line.startswith("UNIQUE-ID"):
                curr_compound_obj.unique_id = str(each_line.split(" - ")[1].strip("\n"))

            # get inchi string ------------------------------
            #  inchi: INCHI - INCHI=1S/C7H15N2O8P/c8-1-4(10)9-7-6(12)5(11)3(17-7)2-16-18(13,14)1
            elif each_line.startswith("INCHI -"):
                curr_compound_obj.inchi = str(each_line.replace('INCHI - ', '')).strip("\n")

            # get inchi key ------------------------------
            #  inchikey: INCHI-KEY - InChIKey=KFWWCMJSYSSPSK-BOGFJHSMSA-J
            elif each_line.startswith("INCHI-KEY"):
                flag_hit_inchi_key = 1
                curr_compound_obj.inchikey = str(each_line.replace('INCHI-KEY - InChIKey=', '')).strip("\n")

            # get smile string ------------------------------
            #  smile: SMILES - CNC(=O)CCC([N+])C(=O)[O-]
            elif each_line.startswith("SMILES - "):
                curr_compound_obj.smiles = str(each_line.replace('SMILES - ', '')).strip("\n")

    if not os.path.isfile(output_path):
        initialize_metacyc_hdf5(output_path)

    with h5py.File(output_path, 'a') as h5:
        dset = h5['compound']
        _arr = np.array(
            data,
            dtype=[('filename', H5PY_STR_TYPE), ('unique_id', H5PY_STR_TYPE), ('common_name', H5PY_STR_TYPE),
                   ('inchikey', H5PY_STR_TYPE), ('inchi', H5PY_STR_TYPE), ('smiles', H5PY_STR_TYPE)]
        )

        if not dset.shape:
            del h5['compound']
            h5.create_dataset('compound', data=_arr, shape=_arr.shape[0], maxshape=(None,))
        else:
            current_length = dset.len()
            dset.resize((current_length + _arr.shape[0]), axis=0)
            dset[current_length:] = _arr

        h5.flush()

    return list_compound_obj


def convert_metacyc_pathways_dat_to_h5(path, output_path, parameters_to_open_file=None):
    LOGGER.debug(f'Read {path}')
    if not isinstance(parameters_to_open_file, dict):
        parameters_to_open_file = dict(encoding='utf8')
    filename = os.path.basename(path)

    if not os.path.isfile(output_path):
        initialize_metacyc_hdf5(output_path)

    str_list_compound_unique_id = ''
    data = []

    with open(path, 'r', **parameters_to_open_file) as f:
        for each_line in f:
            if each_line.startswith("//"):
                data.append((
                    filename,
                    unique_id,
                    common_name,
                    str_list_compound_unique_id
                ))

                # initialize str_list_compound_unique_id
                str_list_compound_unique_id = ''

            # get unique id of the pathway ------------------------------
            #  unique_id: UNIQUE-ID - PWY-6398
            elif each_line.startswith("UNIQUE-ID"):
                unique_id = str(each_line.split(" - ")[1].strip("\n"))

            # get common name of the pathway ------------------------------
            elif each_line.startswith("COMMON-NAME"):
                common_name = str(each_line.split(" - ")[1].strip("\n"))

            # read reaction layout line and get unique id of the compounds related in the pathway -------------
            # reaction layout REACTION-LAYOUT
            elif each_line.startswith("REACTION-LAYOUT"):
                # list_primaries_uniq_id = re.findall("PRIMARIES ([^\)]+)\)", each_line)
                reg_left_primaries = re.compile(r"LEFT-PRIMARIES ([^)]+)\)")
                reg_right_primaries = re.compile(r"RIGHT-PRIMARIES ([^)]+)\)")

                left_primaries_uniq_id_line = ''
                match = reg_left_primaries.search(each_line)
                if match is not None:
                    left_primaries_uniq_id_line = match.group(1)

                right_primaries_uniq_id_line = ''
                match = reg_right_primaries.search(each_line)
                if match is not None:
                    right_primaries_uniq_id_line = match.group(1)

                # for direction -----------------------
                direction = ""
                reg_direction = re.compile(r"DIRECTION:([^)]+)\)")
                match = reg_direction.search(each_line)
                if match is not None:
                    direction = match.group(1)

                LOGGER.debug("direction:", direction)
                LOGGER.debug("left_primaries_uniq_id_line ", left_primaries_uniq_id_line)
                LOGGER.debug("right_primaries_uniq_id_line", right_primaries_uniq_id_line)

                list_left_primaries_uniq_id = left_primaries_uniq_id_line.split()
                list_right_primaries_uniq_id = right_primaries_uniq_id_line.split()

                # append compound unique-id to the list
                # LOGGER.debug "left primaries"
                for cmpd_uniq_id in list_left_primaries_uniq_id:
                    if not str_list_compound_unique_id:
                        str_list_compound_unique_id = cmpd_uniq_id
                    else:
                        str_list_compound_unique_id += f', {cmpd_uniq_id}'

                # LOGGER.debug "right primaries"
                for cmpd_uniq_id in list_right_primaries_uniq_id:
                    if not str_list_compound_unique_id:
                        str_list_compound_unique_id = cmpd_uniq_id
                    else:
                        str_list_compound_unique_id += f', {cmpd_uniq_id}'

                # removing redundant after appending to the list for the object
                # this is not right way/time/place to remove redundant, but it is easy to organize.
                # curr_pathway_obj.list_compound_unique_id = list(set(curr_pathway_obj.list_compound_unique_id))

    with h5py.File(output_path, 'a') as h5:
        dset = h5['pathway']
        _arr = np.array(
            data,
            dtype=[('filename', H5PY_STR_TYPE), ('unique_id', H5PY_STR_TYPE),
                   ('common_name', H5PY_STR_TYPE), ('compound_unique_id_list', H5PY_STR_TYPE)]
        )

        if not dset.shape:
            del h5['pathway']
            h5.create_dataset('pathway', data=_arr, shape=_arr.shape[0], maxshape=(None,))
        else:
            current_length = dset.len()
            dset.resize((current_length + _arr.shape[0]), axis=0)
            dset[current_length:] = _arr

        h5.flush()


def convert_metacyc_pathways_dat_to_npy(path, output_path, parameters_to_open_file=None):
    LOGGER.debug(f'Read {path}')
    if not isinstance(parameters_to_open_file, dict):
        parameters_to_open_file = dict(encoding='utf8')
    filename = os.path.basename(path)

    # Make output folder if it does not exist.
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    dtypes = get_metacyc_pathway_dtype()

    str_list_compound_unique_id = ''
    compound_unique_id_list = []
    data = []

    with open(path, 'r', **parameters_to_open_file) as f:
        for each_line in f:
            if each_line.startswith("//"):

                current_data = []
                for field_name, _ in dtypes:
                    if field_name == 'filename':
                        current_data.append(filename)
                    elif field_name == 'unique_id':
                        current_data.append(unique_id)
                    elif field_name == 'common_name':
                        current_data.append(common_name)
                    elif field_name == 'compound_unique_id_list':
                        current_data.append(compound_unique_id_list)

                data.append(tuple(current_data))

                # initialize compound_unique_id_list
                compound_unique_id_list = []

            # get unique id of the pathway ------------------------------
            #  unique_id: UNIQUE-ID - PWY-6398
            elif each_line.startswith("UNIQUE-ID"):
                unique_id = str(each_line.split(" - ")[1].strip("\n"))

            # get common name of the pathway ------------------------------
            elif each_line.startswith("COMMON-NAME"):
                common_name = str(each_line.split(" - ")[1].strip("\n"))

            # read reaction layout line and get unique id of the compounds related in the pathway -------------
            # reaction layout REACTION-LAYOUT
            elif each_line.startswith("REACTION-LAYOUT"):
                # list_primaries_uniq_id = re.findall("PRIMARIES ([^\)]+)\)", each_line)
                reg_left_primaries = re.compile(r"LEFT-PRIMARIES ([^)]+)\)")
                reg_right_primaries = re.compile(r"RIGHT-PRIMARIES ([^)]+)\)")

                left_primaries_uniq_id_line = ''
                match = reg_left_primaries.search(each_line)
                if match is not None:
                    left_primaries_uniq_id_line = match.group(1)

                right_primaries_uniq_id_line = ''
                match = reg_right_primaries.search(each_line)
                if match is not None:
                    right_primaries_uniq_id_line = match.group(1)

                # for direction -----------------------
                direction = ""
                reg_direction = re.compile(r"DIRECTION:([^)]+)\)")
                match = reg_direction.search(each_line)
                if match is not None:
                    direction = match.group(1)

                LOGGER.debug("direction:", direction)
                LOGGER.debug("left_primaries_uniq_id_line ", left_primaries_uniq_id_line)
                LOGGER.debug("right_primaries_uniq_id_line", right_primaries_uniq_id_line)

                list_left_primaries_uniq_id = [x.strip() for x in left_primaries_uniq_id_line.split() if x.strip()]
                list_right_primaries_uniq_id = [x.strip() for x in right_primaries_uniq_id_line.split() if x.strip()]

                # append compound unique-id to the list
                compound_unique_id_list = list_left_primaries_uniq_id + list_right_primaries_uniq_id

                # removing redundant after appending to the list for the object
                # this is not right way/time/place to remove redundant, but it is easy to organize.
                # curr_pathway_obj.list_compound_unique_id = list(set(curr_pathway_obj.list_compound_unique_id))

    arr = np.array(data, dtype=dtypes)

    # Add already existing array to arr.
    if os.path.isfile(output_path):
        existing_arr = np.load(output_path, allow_pickle=True)
        arr = np.hstack((existing_arr, arr))

    np.save(output_path, arr)


def read_metacyc_pathways_dat(path, output_path, parameters_to_open_file=None):
    if not isinstance(parameters_to_open_file, dict):
        parameters_to_open_file = {}
    filename = os.path.basename(path)

    list_pathway_obj = []  # type: list[MetaCycPathway]

    # need to create first object ----------------------
    curr_pathway_obj = MetaCycPathway()
    # network object as well
    DG_pathway = nx.DiGraph()

    data = []
    with open(path, 'r', **parameters_to_open_file) as f:
        for each_line in f:
            # first line
            if each_line.startswith("//"):
                # end of a pathway info. append it.
                curr_pathway_obj.nx_directed_graph = DG_pathway
                list_pathway_obj.append(curr_pathway_obj)

                data.append((
                    filename,
                    curr_pathway_obj.unique_id,
                    curr_pathway_obj.common_name,
                ))

                # then create new
                curr_pathway_obj = MetaCycPathway()
                # you need create new graph object
                DG_pathway = nx.DiGraph()

            # get unique id of the pathway ------------------------------
            #  unique_id: UNIQUE-ID - PWY-6398
            elif each_line.startswith("UNIQUE-ID"):
                curr_pathway_obj.unique_id = str(each_line.split(" - ")[1].strip("\n"))

            elif each_line.startswith("COMMON-NAME"):
                curr_pathway_obj.common_name = str(each_line.split(" - ")[1].strip("\n"))

            # read reaction layout line and get unique id of the compounds related in the pathway -------------
            # reaction layout REACTION-LAYOUT
            elif each_line.startswith("REACTION-LAYOUT"):
                # list_primaries_uniq_id = re.findall("PRIMARIES ([^\)]+)\)", each_line)
                reg_left_primaries = re.compile(r"LEFT-PRIMARIES ([^)]+)\)")
                reg_right_primaries = re.compile(r"RIGHT-PRIMARIES ([^)]+)\)")

                match = reg_left_primaries.search(each_line)
                if match is not None:
                    left_primaries_uniq_id_line = match.group(1)

                match = reg_right_primaries.search(each_line)
                if match is not None:
                    right_primaries_uniq_id_line = match.group(1)

                # for direction -----------------------
                direction = ""
                reg_direction = re.compile(r"DIRECTION:([^)]+)\)")
                match = reg_direction.search(each_line)
                if match is not None:
                    direction = match.group(1)

                LOGGER.debug("direction:", direction)
                LOGGER.debug("left_primaries_uniq_id_line ", left_primaries_uniq_id_line)
                LOGGER.debug("right_primaries_uniq_id_line", right_primaries_uniq_id_line)

                list_left_primaries_uniq_id = left_primaries_uniq_id_line.split()
                list_right_primaries_uniq_id = right_primaries_uniq_id_line.split()

                # append compound unique-id to the list
                # LOGGER.debug "left primaries"
                for cmpd_uniq_id in list_left_primaries_uniq_id:
                    curr_pathway_obj.list_compound_unique_id.append(cmpd_uniq_id)

                # LOGGER.debug "right primaries"
                for cmpd_uniq_id in list_right_primaries_uniq_id:
                    curr_pathway_obj.list_compound_unique_id.append(cmpd_uniq_id)

                ###############################################################################
                # create network object
                ###############################################################################

                if direction == "L2R":
                    for cmpd_uniq_id_left in list_left_primaries_uniq_id:
                        for cmpd_uniq_id_right in list_right_primaries_uniq_id:
                            LOGGER.debug("DGing")
                            DG_pathway.add_edge(cmpd_uniq_id_left, cmpd_uniq_id_right)

                if direction == "R2L":
                    for cmpd_uniq_id_right in list_right_primaries_uniq_id:
                        for cmpd_uniq_id_left in list_left_primaries_uniq_id:
                            DG_pathway.add_edge(cmpd_uniq_id_right, cmpd_uniq_id_left)

                # removing redundant after appending to the list for the object
                # this is not right way/time/place to remove redundant, but it is easy to organize.
                # curr_pathway_obj.list_compound_unique_id = list(set(curr_pathway_obj.list_compound_unique_id))

    if not os.path.isfile(output_path):
        initialize_metacyc_hdf5(output_path)

    with h5py.File(output_path, 'a') as h5:
        dset = h5['pathway']
        _arr = np.array(
            data,
            dtype=[('filename', H5PY_STR_TYPE), ('unique_id', H5PY_STR_TYPE), ('common_name', H5PY_STR_TYPE)]
        )

        if not dset.shape:
            del h5['pathway']
            h5.create_dataset('pathway', data=_arr, shape=_arr.shape[0], maxshape=(None,))
        else:
            current_length = dset.len()
            dset.resize((current_length + _arr.shape[0]), axis=0)
            dset[current_length:] = _arr

        h5.flush()

    return list_pathway_obj


######################################################################################################################
# assign pathway id and name to compound object
#  list_compound_obj: list of compound object of metacyc made in this class
#  list_pathway_obj: list of pathway object of metacyc made in this class
######################################################################################################################

def assign_pathway_id_to_compound(list_compound_obj, list_pathway_obj):
    for compound_obj in list_compound_obj:

        for pathway_obj in list_pathway_obj:

            for unique_id in pathway_obj.list_compound_unique_id:
                if compound_obj.unique_id == unique_id:
                    compound_obj.list_pathway_unique_id.append(pathway_obj.unique_id)
                    compound_obj.list_pathway_common_name.append(pathway_obj.common_name)

#
#  compound file data from plant cyc look different.
#   basically one line contain all info for one compound like
#   5-PHOSPHORIBOSYL-N-FORMYLGLYCINEAMIDINE	2-(formamido)-N1-(5-phospho-beta-D-ribosyl)acetamidine	5-phosphoribosyl-<i>N</i>-formylglycineamidine*5'-phosphoribosyl-<i>N</i>-formyl glycineamidine*FGAM*5'-phosphoribosylformylglycinamidine		C8 H15 N3 O8 P1 	C(NC=O)C(=[N+])NC1(C(O)C(O)C(COP([O-])(=O)[O-])O1)		EC-6.3.3.1	ATP + 2-(formamido)-N1-(5-phospho-beta-D-ribosyl)acetamidine  ->  ADP + 5-amino-1-(5-phospho-beta-D-ribosyl)imidazole + phosphate + H+	5-aminoimidazole ribonucleotide biosynthesis I


def assign_pathway_id_to_compound_in_h5(path):
    with h5py.File(path, 'a') as h5:
        dset_compound = h5['compound']
        dset_pathway = h5['pathway']

        ser_compound_unique_ids = pd.Series(
            list(map(lambda x: tuple(re.findall(r'[A-Za-z0-9-]+', x)), dset_pathway['compound_unique_id_list'].astype(str)))
        )

        pathway_unique_id_data = []
        pathway_common_name_data = []
        for rec in dset_compound[()]:
            str_pathway_unique_id_list = ''
            str_pathway_common_name_list = ''
            mask = ser_compound_unique_ids.apply(lambda ids: rec['unique_id'] in ids)
            if np.any(mask):
                arr_pathway_unique_id = dset_pathway[mask]['unique_id']
                str_pathway_unique_id_list = ', '.join(arr_pathway_unique_id)

                arr_pathway_common_name = dset_pathway[mask]['common_name']
                str_pathway_common_name_list = ', '.join(arr_pathway_common_name)

            pathway_unique_id_data.append(str_pathway_unique_id_list)
            pathway_common_name_data.append(str_pathway_common_name_list)

        unique_id_arr_to_add = np.array(pathway_unique_id_data).astype('S')
        common_name_arr_to_add = np.array(pathway_common_name_data).astype('S')
        new_compound_arr = rfn.append_fields(dset_compound[()], 'pathway_unique_id_list',
                                             unique_id_arr_to_add, usemask=False)
        new_compound_arr = rfn.append_fields(new_compound_arr, 'pathway_common_name_list',
                                             common_name_arr_to_add, usemask=False)

        del h5['compound']
        h5.create_dataset('compound', data=new_compound_arr, shape=new_compound_arr.shape, maxshape=(None,))
        h5.flush()


def assign_pathway_id_to_compound_in_npy(compound_path, pathway_path):
    if (not os.path.isfile(compound_path)) or (not os.path.isfile(pathway_path)):
        return

    # Load compound array
    arr_compound = np.load(compound_path, allow_pickle=True)
    if not arr_compound.size:
        return

    # Load pathway array
    arr_pathway = np.load(pathway_path, allow_pickle=True)
    if not arr_pathway.size:
        return

    ser_compound_unique_ids = pd.Series(arr_pathway['compound_unique_id_list'])

    # Get unique_id and common_name of pathway.
    pathway_unique_id_data = []
    pathway_common_name_data = []
    for rec in arr_compound:
        pathway_unique_id_list = []
        pathway_common_name_list = []
        mask = ser_compound_unique_ids.apply(lambda ids: rec['unique_id'] in ids)
        if np.any(mask):
            arr_pathway_unique_id = arr_pathway['unique_id'][mask]
            pathway_unique_id_list = list(arr_pathway_unique_id)

            arr_pathway_common_name = arr_pathway['common_name'][mask]
            pathway_common_name_list = list(arr_pathway_common_name)

        pathway_unique_id_data.append(pathway_unique_id_list)
        pathway_common_name_data.append(pathway_common_name_list)

    # Assign pathway_unique_id_data and pathway_common_name_data to arr_compound
    arr_compound['pathway_unique_id_list'] = pathway_unique_id_data
    arr_compound['pathway_common_name_list'] = pathway_common_name_data

    # Update array file.
    np.save(compound_path, arr_compound)


def read_plantcyc_compounds(path, parameters_to_open_file=None):
    if not isinstance(parameters_to_open_file, dict):
        parameters_to_open_file = {}

    list_compound_obj = []  # type: list[MetaCycCompound]

    with open(path, 'r', **parameters_to_open_file) as f:
        for each_line in f:
            list_tab = each_line.split("\t")

            if len(list_tab) > 9:
                # need to create first object
                curr_compound_obj = MetaCycCompound()

                curr_compound_obj.unique_id = str(each_line.split("\t")[0].strip("\n"))
                curr_compound_obj.common_name = str(each_line.split("\t")[1].strip("\n"))
                curr_compound_obj.smiles = str(each_line.split("\t")[5].strip("\n"))

                curr_compound_obj.ec = str(each_line.split("\t")[7].strip("\n"))
                curr_compound_obj.reaction_equation = str(each_line.split("\t")[8].strip("\n"))
                curr_compound_obj.pathway = str(each_line.split("\t")[9].strip("\n"))

                my_mol = Chem.MolFromSmiles(curr_compound_obj.smiles)

                # the compound file contains non compound lines "RXN" etc.
                # so we use rdkit mol validity to see the line is compound or not

                if my_mol is not None:
                    my_inchi = Chem.MolToInchi(my_mol)
                    curr_compound_obj.inchi = my_inchi
                    curr_compound_obj.inchikey = Chem.InchiToInchiKey(curr_compound_obj.inchi)

                    list_compound_obj.append(curr_compound_obj)

    # make curr_compound_obj non-redundant/
    # use unique ID as key.
    set_seen_id = set()
    list_compound_obj_nr = []

    for o in list_compound_obj:

        if o.unique_id not in set_seen_id:
            LOGGER.debug("making non-redundant", o.inchikey)
            list_compound_obj_nr.append(o)
            set_seen_id.add(o.unique_id)

    return list_compound_obj_nr
