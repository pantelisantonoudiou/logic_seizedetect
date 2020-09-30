# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:04:58 2020

@author: panton01
"""


import os, json


def main_func():
    """
    Get path and save settings from user

    Returns
    -------
    bool, False if operation fails.

    """
    # get user input
    path = input('Enter Path for analysis: \n')
    
    # check if path exists
    if os.path.isdir(path) == 0: # if path exists
        print('\n************ The input', '"'+ path +'"' ,'is not a valid path. Please try again ************\n')
        return False
    
    # create dict for export
    to_export = {'main_path':path, 'not_analyzed_only':0}
    
    # Load config file 
    try:
        open('config.json', 'w').write(json.dumps(to_export))
    except Exception as err:
        print('Unable to write the config file.\n',err)

    print('\n------> Settings saved to config.json <------\n')
    
# Execute if module runs as main program
if __name__ == '__main__' :
    main_func() # run function 