import os
from jumeg.jumeg_preprocessing import get_files_from_list
def reset_directory(path=None):
    """
    check whether the directory exits, if yes, recreat the directory
    ----------
    path : the target directory.
    """
    path_list = get_files_from_list(path)
    # loop across all filenames
    for fn_path in path_list:
        import shutil
        isexists = os.path.exists(fn_path)
        if isexists:
            shutil.rmtree(fn_path)
        os.makedirs(fn_path)


def set_directory(path=None):
    """
    check whether the directory exits, if no, creat the directory
    ----------
    path : the target directory.

    """
    path_list = get_files_from_list(path)
    # loop across all filenames
    for fn_path in path_list:
        isexists = os.path.exists(fn_path)
        if not isexists:
            os.makedirs(fn_path)