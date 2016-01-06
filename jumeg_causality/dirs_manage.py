import os
def reset_directory(path=None):
    """
    check whether the directory exits, if yes, recreat the directory
    ----------
    path : the target directory.
    """
    import shutil
    isexists = os.path.exists(path)
    if isexists:
        shutil.rmtree(path)
    os.makedirs(path)


def set_directory(path=None):
    """
    check whether the directory exits, if no, creat the directory
    ----------
    path : the target directory.

    """
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)