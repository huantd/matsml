# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# This file is auto-generated by substrates/meta/rewrite.py
# It will be surfaced by the build system as a symlink at:
#   `tensorflow_probability/substrates/jax/experimental/parallel_filter/__init__.py`
# For more info, see substrate_runfiles_symlinks in build_defs.bzl
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# (This notice adds 10 to line numbering.)


# Copyright 2020 The TensorFlow Probability Authors.
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
"""TensorFlow Probability experimental parallel filtering package."""
from tensorflow_probability.substrates.jax.experimental.parallel_filter.parallel_kalman_filter_lib import kalman_filter
from tensorflow_probability.substrates.jax.experimental.parallel_filter.parallel_kalman_filter_lib import sample_walk
from tensorflow_probability.python.internal import all_util


_allowed_symbols = ['kalman_filter',
                    'sample_walk']

all_util.remove_undocumented(__name__, _allowed_symbols)

