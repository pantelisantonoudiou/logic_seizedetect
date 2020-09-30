# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:59:58 2020

@author: panton01
"""

from pick import pick
title = 'Please choose your favorite programming language: '
options = ['Java', 'JavaScript', 'Python', 'PHP', 'C++', 'Erlang', 'Haskell']
option, index = pick(options, title)
print(option)
print(index)