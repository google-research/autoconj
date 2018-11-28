# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Various utility functions and classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

import autograd
from autograd import numpy as np
from autograd.numpy import numpy_boxes, numpy_vspaces
import autograd.util as ag_util

import numpy as onp


def Enum(name, fields):
  return type(name, (), dict(zip(map(str.upper, fields), itertools.count())))


SupportTypes = Enum('SupportTypes', ['REAL', 'NONNEGATIVE', 'UNIT_INTERVAL',
                                     'SIMPLEX', 'BINARY', 'INTEGER', 'ONE_HOT'])
support_type_to_name = {i: name for name, i in SupportTypes.__dict__.items()
                        if name[0] != '_'}



def split_einsum_formula(formula):
  joined_input_formulas, output_formula = formula.split('->')
  return joined_input_formulas.split(','), output_formula


# Monkey-patch to register int and boolean types with ArrayBox.
int_types = [int, onp.int16, onp.int32, onp.int64]
bool_types = [bool, onp.bool_]
for type_ in int_types + bool_types:
  numpy_boxes.ArrayBox.register(type_)
  numpy_vspaces.ArrayVSpace.register(type_)
