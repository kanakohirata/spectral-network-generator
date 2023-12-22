import os


def get_specific_ext_paths(dir_path, ext) -> list:
    """
    Parameters
    ----------
    dir_path : str
    ext : str

    Returns
    -------
    list
        If dir_path includes 'file0.txt', 'file1.tsv', 'file2.txt' and 'file3.txt' files and ext is '.txt',
        the following list will be returned.

        list[path_to_file0.txt,
             path_to_file2.txt,
             path_to_file3.txt]
    """
    paths = []
    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and os.path.splitext(filename)[1] == ext:
            paths.append(path)

    return paths


def get_npy_paths(dir_path, ext='.npy') -> list:
    """
    Parameters
    ----------
    dir_path : str
    ext : str

    Returns
    -------
    list
        If dir_path includes 'file0.npy', 'file1.tsv', 'file2.npy' and 'file3.npy' files and ext is '.npy',
        the following list will be returned.

        list[path_to_file0.npy,
             path_to_file2.npy,
             path_to_file3.npy]
    """
    return get_specific_ext_paths(dir_path, ext=ext)


def get_folder_name_list(parent_folder_path):
    """
    Parameters
    ----------
    parent_folder_path : str

    Returns
    -------
    list
        If parent_folder_path includes 'folder_0', 'folder_1', 'folder_2' folders,
        the following list will be returned.

        list['folder_0', 'folder_1', 'folder_2']
    """
    dir_names = []
    for name in os.listdir(parent_folder_path):
        path = os.path.join(parent_folder_path, name)

        if os.path.isdir(path):
            dir_names.append(name)

    return dir_names


def get_folder_name_vs_path_list(parent_folder_path):
    """
    Parameters
    ----------
    parent_folder_path : str

    Returns
    -------
    list
        If parent_folder_path includes 'folder_0', 'folder_1', 'folder_2' folders,
        the following list will be returned.

        list[('folder_0', path_to_folder_0),
             ('folder_1', path_to_folder_1),
             ('folder_2', path_to_folder_2)]
    """
    name_vs_path_list = []
    for name in os.listdir(parent_folder_path):
        path = os.path.join(parent_folder_path, name)

        if os.path.isdir(path):
            name_vs_path_list.append((name, path))

    return name_vs_path_list


def get_folder_name_vs_path_dict(parent_folder_path) -> dict:
    """
    Parameters
    ----------
    parent_folder_path : str

    Returns
    -------
    dict
        If parent_folder_path includes 'folder_0', 'folder_1', 'folder_2' folders,
        the following dict will be returned.
        dict{'folder_0': path_to_folder_0,
             'folder_1': path_to_folder_1,
             'folder_2': path_to_folder_2,}
    """
    dir_name_vs_path_dict = {}

    for name in os.listdir(parent_folder_path):
        path = os.path.join(parent_folder_path, name)

        if os.path.isdir(path):
            dir_name_vs_path_dict[name] = path

    return dir_name_vs_path_dict
