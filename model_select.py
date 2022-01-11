# -*- coding: utf-8 -*-


from model_selection.test_models import MethodTest

path = r'\\SUPERCOMPUTER2\Shared\seizure_detect_data\train'
obj = MethodTest(path)
obj.multi_folder() # get catalogue for multiple folders
