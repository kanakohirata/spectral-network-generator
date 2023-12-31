import os
from . import grouping_metadata
from .grouping_spectra import group_spectra


if os.path.isfile(grouping_metadata.SAMPLE_DATASET_KEYWORDS_PATH):
    os.remove(grouping_metadata.SAMPLE_DATASET_KEYWORDS_PATH)

if os.path.isfile(grouping_metadata.REF_DATASET_KEYWORDS_PATH):
    os.remove(grouping_metadata.REF_DATASET_KEYWORDS_PATH)

if os.path.isfile(grouping_metadata.PATH_OF_SAMPLE_DATASET_KEYWORDS_VS_LABEL_DICT):
    os.remove(grouping_metadata.PATH_OF_SAMPLE_DATASET_KEYWORDS_VS_LABEL_DICT)

if os.path.isfile(grouping_metadata.PATH_OF_REF_DATASET_KEYWORDS_VS_LABEL_DICT):
    os.remove(grouping_metadata.PATH_OF_REF_DATASET_KEYWORDS_VS_LABEL_DICT)
