import os
from . import grouping_metadata


if os.path.isfile(grouping_metadata.SAMPLE_DATASET_KEYWORDS_PATH):
    os.remove(grouping_metadata.SAMPLE_DATASET_KEYWORDS_PATH)

if os.path.isfile(grouping_metadata.REF_DATASET_KEYWORDS_PATH):
    os.remove(grouping_metadata.REF_DATASET_KEYWORDS_PATH)

