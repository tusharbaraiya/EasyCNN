import glob


# function to generate list of subdirectories
def get_sub_dirs(path):
    return glob.glob(path + "/*/")
