import os
import shutil
from utils import get_paths


def initialize_score_files(dir_path='./scores',
                           clustered_dir_name='clustered',
                           grouped_dir_name='grouped',
                           raw_dir_name='raw'):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    else:
        for f in os.listdir(dir_path):
            p = os.path.join(dir_path, f)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.splitext(f)[1] == '.npy':
                os.remove(p)

    dir_path_raw = os.path.join(dir_path, raw_dir_name)
    if not os.path.isdir(dir_path_raw):
        os.makedirs(dir_path_raw)

    dir_path_clustered = os.path.join(dir_path, clustered_dir_name)
    if not os.path.isdir(dir_path_clustered):
        os.makedirs(dir_path_clustered)

    dir_path_grouped = os.path.join(dir_path, grouped_dir_name)
    if not os.path.isdir(dir_path_grouped):
        os.makedirs(dir_path_grouped)

                
def save_ref_score_file(output_dir):
    ref_clustered_score_dirs = get_paths.get_dirs_of_clustered_score_for_inner_ref()
    ref_clustered_score_dirs += get_paths.get_dirs_of_clustered_score_for_inter_ref()

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for ref_clustered_score_dir in ref_clustered_score_dirs:
        folder_name = os.path.basename(ref_clustered_score_dir)
        dir_to_copy = os.path.join(output_dir, folder_name)
        shutil.copytree(ref_clustered_score_dir, dir_to_copy)

