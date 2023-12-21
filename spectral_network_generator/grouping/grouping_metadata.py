import numpy as np
import os
import pandas as pd
import pickle


SAMPLE_DATASET_KEYWORDS_PATH = './__sample_dataset_keywords.pickle'
REF_DATASET_KEYWORDS_PATH = './__ref_dataset_keywords.pickle'


def _add_dataset_keyword():
    pass


def group_sample_by_dataset(sample_metadata_path, output_dir, split_category,
                            output_filename_prefix='sample_dataset_', export_tsv=False):
    """

    Parameters
    ----------
    sample_metadata_path : str
    output_dir : str
    split_category :str
        'tag' or 'source_filename'
    output_filename_prefix : str
    export_tsv : bool

    Returns
    -------

    """
    if split_category not in ('tag', 'source_filename'):
        raise ValueError('Unacceptable split key.')

    dataset_keyword_list = []
    if os.path.isfile(SAMPLE_DATASET_KEYWORDS_PATH):
        with open(SAMPLE_DATASET_KEYWORDS_PATH, 'rb') as f:
            dataset_keyword_list = pickle.load(f)

    # Make the output folder if it does not exist.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    arr = np.load(sample_metadata_path, allow_pickle=True)
    keyword_set = np.unique(arr[split_category])

    for keyword in keyword_set:
        dataset_arr = arr[arr[split_category] == keyword]
        # Add dataset keyword
        dataset_arr['keyword'] = keyword

        if keyword not in dataset_keyword_list:
            dataset_keyword_list.append(keyword)

        dataset_idx = dataset_keyword_list.index(keyword)
        output_filename = f'{output_filename_prefix}{dataset_idx}.npy'
        output_path = os.path.join(output_dir, output_filename)

        # Add dataset_arr to already existing dataset array.
        if os.path.isfile(output_path):
            existing_arr = np.load(output_path, allow_pickle=True)
            dataset_arr = np.hstack((existing_arr, dataset_arr))

        with open(output_path, 'wb') as f:
            np.save(f, dataset_arr)
            f.flush()

        if export_tsv:
            tsv_path = os.path.splitext(output_path)[0] + '.tsv'
            df = pd.DataFrame.from_records(dataset_arr)
            df.to_csv(tsv_path, sep='\t', index=False)

    with open(SAMPLE_DATASET_KEYWORDS_PATH, 'wb') as f:
        pickle.dump(dataset_keyword_list, f)
        f.flush()


def group_reference_by_dataset(ref_metadata_path, output_dir, split_category,
                               output_filename_prefix='ref_dataset_', export_tsv=False):
    """

    Parameters
    ----------
    ref_metadata_path : str
    output_dir : str
    split_category : str
        'tag', 'source_filename', 'cmpd_classification_superclass', 'cmpd_classification_class' or 'cmpd_pathway'
    output_filename_prefix : str
    export_tsv : bool

    Returns
    -------

    """
    if split_category not in ('tag', 'source_filename', 'cmpd_classification_superclass', 'cmpd_classification_class',
                              'cmpd_pathway'):
        raise ValueError('Unacceptable split key.')

    dataset_keyword_list = []
    if os.path.isfile(REF_DATASET_KEYWORDS_PATH):
        with open(REF_DATASET_KEYWORDS_PATH, 'rb') as f:
            dataset_keyword_list = pickle.load(f)

    # Make the output folder if it does not exist.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Load metadata array.
    arr = np.load(ref_metadata_path, allow_pickle=True)

    # Get unique dataset keywords. -----------------------------------------------------------------------------------
    if split_category in ('tag', 'source_filename', 'cmpd_classification_superclass', 'cmpd_classification_class'):
        # If category is '', it will be replaced with 'no_category'
        category_arr = np.array(arr[split_category])
        category_arr[category_arr == ''] = 'no_category'
        keyword_set = np.unique(category_arr)

    else:
        # If category is [], it will be replaced with ['no_category']
        category_arr = np.array(arr['pathway_common_name_list'])
        category_arr[category_arr == []] = ['no_category']

        _keyword_list = []
        for keywords in category_arr:
            _keyword_list.extend(keywords)

        keyword_set = np.unique(_keyword_list)
    # ----------------------------------------------------------------------------------------------------------------

    # Divide spectra by dataset.
    for keyword in keyword_set:
        if split_category in ('tag', 'source_filename', 'cmpd_classification_superclass', 'cmpd_classification_class'):
            dataset_arr = arr[category_arr == keyword]
        else:
            dataset_mask = [keyword in keywords for keywords in category_arr]
            dataset_arr = arr[dataset_mask]

        # Add dataset keyword
        dataset_arr['keyword'] = keyword

        if keyword not in dataset_keyword_list:
            dataset_keyword_list.append(keyword)

        dataset_idx = dataset_keyword_list.index(keyword)
        output_filename = f'{output_filename_prefix}{dataset_idx}.npy'
        output_path = os.path.join(output_dir, output_filename)

        # Add dataset_arr to already existing dataset array.
        if os.path.isfile(output_path):
            existing_arr = np.load(output_path, allow_pickle=True)
            dataset_arr = np.hstack((existing_arr, dataset_arr))

        with open(output_path, 'wb') as f:
            np.save(f, dataset_arr)
            f.flush()

        if export_tsv:
            tsv_path = os.path.splitext(output_path)[0] + '.tsv'
            df = pd.DataFrame.from_records(dataset_arr)
            df.to_csv(tsv_path, sep='\t', index=False)

    with open(SAMPLE_DATASET_KEYWORDS_PATH, 'wb') as f:
        pickle.dump(dataset_keyword_list, f)
        f.flush()
