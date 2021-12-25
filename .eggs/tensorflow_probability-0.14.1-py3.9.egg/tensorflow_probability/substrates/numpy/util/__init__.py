# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""TensorFlow Probability python utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.internal import all_util
from tensorflow_probability.substrates.numpy.internal.parameter_properties import ParameterProperties
from tensorflow_probability.substrates.numpy.util.deferred_tensor import DeferredTensor
from tensorflow_probability.substrates.numpy.util.deferred_tensor import TransformedVariable
from tensorflow_probability.substrates.numpy.util.seed_stream import SeedStream


_allowed_symbols = [
    'DeferredTensor',
    'ParameterProperties',
    'TransformedVariable',
    'SeedStream',
]

all_util.remove_undocumented(__name__, _allowed_symbols)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# This file is auto-generated by substrates/meta/rewrite.py
# It will be surfaced by the build system as a symlink at:
#   `tensorflow_probability/substrates/numpy/util/__init__.py`
# For more info, see substrate_runfiles_symlinks in build_defs.bzl
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@