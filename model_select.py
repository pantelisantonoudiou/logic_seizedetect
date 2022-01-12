# -*- coding: utf-8 -*-


# from model_selection.model_metrics_train import MethodTest
from model_selection.model_metrics_test import MethodTest

path = r'\\SUPERCOMPUTER2\Shared\seizure_detect_data\test'
obj = MethodTest(path)
obj.multi_folder() # get catalogue for multiple folders
