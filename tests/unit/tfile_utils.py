# Copyright (C) 2020 NRL
# Author: Angeline Burrell

from glob import glob
import os


def get_test_files(test_file_type, test_dir=os.path.join("..", "testfiles")):
    """ Generate a dictionary containing the test filenames

    Parameters
    ----------
    test_file_type : str
        Accepts "good", "stream", and "corrupt"
    test_dir : str
        Directory containing the test files
        (default=os.path.join('..', 'testfiles'))

    Returns
    -------
    test_files : list or dict
        Dict of good files with keys pertaining to the file type, a list
        of corrupt files, or a list of stream files

    """
    # Ensure the test file type is lowercase
    test_file_type = test_file_type.lower()

    # Get a list of the available test files
    files = glob("ls {:s}".format(os.path.join(test_dir, test_file_type, "*")))

    # Prepare the test files in the necessary output format
    if test_file_type == "good":
        test_files = {fname.split(".")[-1]: fname for fname in files}
    else:
        test_files = files

    return test_files
        
