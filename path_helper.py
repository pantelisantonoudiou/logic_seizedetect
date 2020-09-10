# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:54:03 2020

@author: panton01
"""
import os,sys
import numpy as np

def get_dir(instr,pathsup):
    """
    folder = get_dir(instr,pathsup)
    folder = get_dir(s,1) # get the inner most folder

    Parameters
    ----------
    instr : STRING
        PATH.
    pathsup : INTEGER
        Defines the number of folders away from the innermost.

    Returns
    -------
    oustr: STRING
        Folder name.

    """
    x = instr.split(os.sep) # split string based on operating system
    oustr = x[-pathsup] #
    return oustr
    

def sep_dir(instr,sepint):
    """
    first_path, inner_dir = sep_dir(instr,sepint)
    first_path, inner_dir = sep_dir(s,1)

    Parameters
    ----------
    instr : STRING
        PATH.
    sepint : INTEGER
         Defines the number of folders away from the innermost.

    Returns
    -------
    first_path : STRING
        first part of path.
    inner_dir : STRING
        second part of path.

    """
    
    # get path separator
    separator = os.sep
    
    # split based on operating system
    x = instr.split(separator)
    
    # get inner directory
    inner_dir = separator.join(x[-sepint:])

    # join based on separator
    first_path =  separator.join(x[0:-sepint])
    
    return first_path, inner_dir
    

def rem_array(start,stop,div):
    """
    rem_array(start,stop,div)
    idx =rem_array(start,stop,div)
    
    Parameters
    ----------
    start : INT
        DESCRIPTION.
    stop : INT
        DESCRIPTION.
    div : INT
        DESCRIPTION.

    Returns
    -------
    idx_array : numpy array
        DESCRIPTION.

    """

    rem = stop % div
    trim_stop = (stop - rem)
    idx_array = np.linspace(start, trim_stop, round(trim_stop/div)+1,dtype = int)
    idx_array = np.append(idx_array, stop)
    
    return idx_array


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")