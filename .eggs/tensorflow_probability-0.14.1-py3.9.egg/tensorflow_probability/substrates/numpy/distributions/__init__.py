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
"""Statistical distributions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,g-importing-member,g-bad-import-order

# Distributions:
from tensorflow_probability.substrates.numpy.distributions.autoregressive import Autoregressive
from tensorflow_probability.substrates.numpy.distributions.batch_broadcast import BatchBroadcast
from tensorflow_probability.substrates.numpy.distributions.batch_concat import BatchConcat
from tensorflow_probability.substrates.numpy.distributions.batch_reshape import BatchReshape
from tensorflow_probability.substrates.numpy.distributions.bates import Bates
from tensorflow_probability.substrates.numpy.distributions.bernoulli import Bernoulli
from tensorflow_probability.substrates.numpy.distributions.beta import Beta
from tensorflow_probability.substrates.numpy.distributions.beta_binomial import BetaBinomial
from tensorflow_probability.substrates.numpy.distributions.beta_quotient import BetaQuotient
from tensorflow_probability.substrates.numpy.distributions.binomial import Binomial
from tensorflow_probability.substrates.numpy.distributions.blockwise import Blockwise
from tensorflow_probability.substrates.numpy.distributions.categorical import Categorical
from tensorflow_probability.substrates.numpy.distributions.cauchy import Cauchy
from tensorflow_probability.substrates.numpy.distributions.chi import Chi
from tensorflow_probability.substrates.numpy.distributions.chi2 import Chi2
from tensorflow_probability.substrates.numpy.distributions.cholesky_lkj import CholeskyLKJ
from tensorflow_probability.substrates.numpy.distributions.continuous_bernoulli import ContinuousBernoulli
from tensorflow_probability.substrates.numpy.distributions.deterministic import Deterministic
from tensorflow_probability.substrates.numpy.distributions.deterministic import VectorDeterministic
from tensorflow_probability.substrates.numpy.distributions.dirichlet import Dirichlet
from tensorflow_probability.substrates.numpy.distributions.dirichlet_multinomial import DirichletMultinomial
from tensorflow_probability.substrates.numpy.distributions.distribution import Distribution
from tensorflow_probability.substrates.numpy.distributions.doublesided_maxwell import DoublesidedMaxwell
from tensorflow_probability.substrates.numpy.distributions.dpp import DeterminantalPointProcess
from tensorflow_probability.substrates.numpy.distributions.empirical import Empirical
from tensorflow_probability.substrates.numpy.distributions.exponentially_modified_gaussian import ExponentiallyModifiedGaussian
from tensorflow_probability.substrates.numpy.distributions.exp_gamma import ExpGamma
from tensorflow_probability.substrates.numpy.distributions.exp_gamma import ExpInverseGamma
from tensorflow_probability.substrates.numpy.distributions.exponential import Exponential
from tensorflow_probability.substrates.numpy.distributions.finite_discrete import FiniteDiscrete
from tensorflow_probability.substrates.numpy.distributions.gamma import Gamma
from tensorflow_probability.substrates.numpy.distributions.gamma_gamma import GammaGamma
from tensorflow_probability.substrates.numpy.distributions.gaussian_process import GaussianProcess
from tensorflow_probability.substrates.numpy.distributions.gaussian_process_regression_model import GaussianProcessRegressionModel
from tensorflow_probability.substrates.numpy.distributions.generalized_normal import GeneralizedNormal
from tensorflow_probability.substrates.numpy.distributions.generalized_pareto import GeneralizedPareto
from tensorflow_probability.substrates.numpy.distributions.geometric import Geometric
from tensorflow_probability.substrates.numpy.distributions.gumbel import Gumbel
from tensorflow_probability.substrates.numpy.distributions.gev import GeneralizedExtremeValue
from tensorflow_probability.substrates.numpy.distributions.half_cauchy import HalfCauchy
from tensorflow_probability.substrates.numpy.distributions.half_normal import HalfNormal
from tensorflow_probability.substrates.numpy.distributions.half_student_t import HalfStudentT
from tensorflow_probability.substrates.numpy.distributions.hidden_markov_model import HiddenMarkovModel
from tensorflow_probability.substrates.numpy.distributions.horseshoe import Horseshoe
from tensorflow_probability.substrates.numpy.distributions.independent import Independent
from tensorflow_probability.substrates.numpy.distributions.inverse_gamma import InverseGamma
from tensorflow_probability.substrates.numpy.distributions.inverse_gaussian import InverseGaussian
from tensorflow_probability.substrates.numpy.distributions.johnson_su import JohnsonSU
from tensorflow_probability.substrates.numpy.distributions.joint_distribution import JointDistribution
from tensorflow_probability.substrates.numpy.distributions.joint_distribution_coroutine import JointDistributionCoroutine
from tensorflow_probability.substrates.numpy.distributions.joint_distribution_auto_batched import JointDistributionCoroutineAutoBatched
from tensorflow_probability.substrates.numpy.distributions.joint_distribution_named import JointDistributionNamed
from tensorflow_probability.substrates.numpy.distributions.joint_distribution_auto_batched import JointDistributionNamedAutoBatched
from tensorflow_probability.substrates.numpy.distributions.joint_distribution_sequential import JointDistributionSequential
from tensorflow_probability.substrates.numpy.distributions.joint_distribution_auto_batched import JointDistributionSequentialAutoBatched
from tensorflow_probability.substrates.numpy.distributions.kumaraswamy import Kumaraswamy
from tensorflow_probability.substrates.numpy.distributions.lambertw_f import LambertWDistribution
from tensorflow_probability.substrates.numpy.distributions.lambertw_f import LambertWNormal
from tensorflow_probability.substrates.numpy.distributions.laplace import Laplace
from tensorflow_probability.substrates.numpy.distributions.linear_gaussian_ssm import LinearGaussianStateSpaceModel
from tensorflow_probability.substrates.numpy.distributions.lkj import LKJ
from tensorflow_probability.substrates.numpy.distributions.logistic import Logistic
from tensorflow_probability.substrates.numpy.distributions.loglogistic import LogLogistic
from tensorflow_probability.substrates.numpy.distributions.lognormal import LogNormal
from tensorflow_probability.substrates.numpy.distributions.logitnormal import LogitNormal
from tensorflow_probability.substrates.numpy.distributions.markov_chain import MarkovChain
from tensorflow_probability.substrates.numpy.distributions.masked import Masked
from tensorflow_probability.substrates.numpy.distributions.matrix_normal_linear_operator import MatrixNormalLinearOperator
from tensorflow_probability.substrates.numpy.distributions.matrix_t_linear_operator import MatrixTLinearOperator
# from tensorflow_probability.substrates.numpy.distributions.mixture import Mixture
from tensorflow_probability.substrates.numpy.distributions.mixture_same_family import MixtureSameFamily
from tensorflow_probability.substrates.numpy.distributions.moyal import Moyal
from tensorflow_probability.substrates.numpy.distributions.multinomial import Multinomial
from tensorflow_probability.substrates.numpy.distributions.multivariate_student_t import MultivariateStudentTLinearOperator
from tensorflow_probability.substrates.numpy.distributions.mvn_diag import MultivariateNormalDiag
from tensorflow_probability.substrates.numpy.distributions.mvn_diag_plus_low_rank import MultivariateNormalDiagPlusLowRank
from tensorflow_probability.substrates.numpy.distributions.mvn_diag_plus_low_rank_covariance import MultivariateNormalDiagPlusLowRankCovariance
from tensorflow_probability.substrates.numpy.distributions.mvn_full_covariance import MultivariateNormalFullCovariance
from tensorflow_probability.substrates.numpy.distributions.mvn_linear_operator import MultivariateNormalLinearOperator
from tensorflow_probability.substrates.numpy.distributions.mvn_tril import MultivariateNormalTriL
from tensorflow_probability.substrates.numpy.distributions.negative_binomial import NegativeBinomial
from tensorflow_probability.substrates.numpy.distributions.normal import Normal
from tensorflow_probability.substrates.numpy.distributions.normal_inverse_gaussian import NormalInverseGaussian
from tensorflow_probability.substrates.numpy.distributions.onehot_categorical import OneHotCategorical
from tensorflow_probability.substrates.numpy.distributions.ordered_logistic import OrderedLogistic
from tensorflow_probability.substrates.numpy.distributions.pareto import Pareto
from tensorflow_probability.substrates.numpy.distributions.pert import PERT
# from tensorflow_probability.substrates.numpy.distributions.pixel_cnn import PixelCNN
from tensorflow_probability.substrates.numpy.distributions.plackett_luce import PlackettLuce
from tensorflow_probability.substrates.numpy.distributions.poisson import Poisson
from tensorflow_probability.substrates.numpy.distributions.poisson_lognormal import PoissonLogNormalQuadratureCompound
from tensorflow_probability.substrates.numpy.distributions.power_spherical import PowerSpherical
from tensorflow_probability.substrates.numpy.distributions.probit_bernoulli import ProbitBernoulli
from tensorflow_probability.substrates.numpy.distributions.quantized_distribution import QuantizedDistribution
from tensorflow_probability.substrates.numpy.distributions.relaxed_bernoulli import RelaxedBernoulli
from tensorflow_probability.substrates.numpy.distributions.relaxed_onehot_categorical import ExpRelaxedOneHotCategorical
from tensorflow_probability.substrates.numpy.distributions.relaxed_onehot_categorical import RelaxedOneHotCategorical
from tensorflow_probability.substrates.numpy.distributions.sample import Sample
from tensorflow_probability.substrates.numpy.distributions.sigmoid_beta import SigmoidBeta
from tensorflow_probability.substrates.numpy.distributions.sinh_arcsinh import SinhArcsinh
from tensorflow_probability.substrates.numpy.distributions.skellam import Skellam
from tensorflow_probability.substrates.numpy.distributions.spherical_uniform import SphericalUniform
from tensorflow_probability.substrates.numpy.distributions.stopping_ratio_logistic import StoppingRatioLogistic
from tensorflow_probability.substrates.numpy.distributions.student_t import StudentT
from tensorflow_probability.substrates.numpy.distributions.student_t_process import StudentTProcess
from tensorflow_probability.substrates.numpy.distributions.transformed_distribution import TransformedDistribution
from tensorflow_probability.substrates.numpy.distributions.triangular import Triangular
from tensorflow_probability.substrates.numpy.distributions.truncated_cauchy import TruncatedCauchy
from tensorflow_probability.substrates.numpy.distributions.truncated_normal import TruncatedNormal
from tensorflow_probability.substrates.numpy.distributions.uniform import Uniform
from tensorflow_probability.substrates.numpy.distributions.variational_gaussian_process import VariationalGaussianProcess
from tensorflow_probability.substrates.numpy.distributions.vector_exponential_diag import VectorExponentialDiag
from tensorflow_probability.substrates.numpy.distributions.von_mises import VonMises
from tensorflow_probability.substrates.numpy.distributions.von_mises_fisher import VonMisesFisher
from tensorflow_probability.substrates.numpy.distributions.weibull import Weibull
from tensorflow_probability.substrates.numpy.distributions.wishart import WishartLinearOperator
from tensorflow_probability.substrates.numpy.distributions.wishart import WishartTriL
from tensorflow_probability.substrates.numpy.distributions.zipf import Zipf

# Utilities/Other:
from tensorflow_probability.substrates.numpy.distributions.joint_distribution_util import independent_joint_distribution_from_structure
from tensorflow_probability.substrates.numpy.distributions.kullback_leibler import RegisterKL
from tensorflow_probability.substrates.numpy.distributions.kullback_leibler import kl_divergence
from tensorflow_probability.substrates.numpy.distributions.normal_conjugate_posteriors import mvn_conjugate_linear_update
from tensorflow_probability.substrates.numpy.distributions.normal_conjugate_posteriors import normal_conjugates_known_scale_posterior
from tensorflow_probability.substrates.numpy.distributions.normal_conjugate_posteriors import normal_conjugates_known_scale_predictive
from tensorflow_probability.substrates.numpy.distributions.poisson_lognormal import quadrature_scheme_lognormal_gauss_hermite
from tensorflow_probability.substrates.numpy.distributions.poisson_lognormal import quadrature_scheme_lognormal_quantiles
from tensorflow_probability.python.internal.reparameterization import FULLY_REPARAMETERIZED
from tensorflow_probability.python.internal.reparameterization import NOT_REPARAMETERIZED
from tensorflow_probability.python.internal.reparameterization import ReparameterizationType

# Module management:
from tensorflow_probability.substrates.numpy.distributions.kullback_leibler import augment_kl_xent_docs
from tensorflow_probability.python.internal.backend.numpy import deprecation  # pylint: disable=g-direct-tensorflow-import


import sys as _sys  # pylint: disable=g-import-not-at-top
augment_kl_xent_docs(_sys.modules[__name__])
del augment_kl_xent_docs
del _sys

# pylint: enable=unused-import,wildcard-import,g-importing-member,g-bad-import-order
# pylint: enable=line-too-long

__all__ = [
    'Autoregressive',
    'BatchBroadcast',
    'BatchReshape',
    'Bates',
    'Bernoulli',
    'Beta',
    'BetaBinomial',
    'BetaQuotient',
    'Binomial',
    'Blockwise',
    'Categorical',
    'Cauchy',
    'Chi',
    'Chi2',
    'CholeskyLKJ',
    'DeterminantalPointProcess',
    'Deterministic',
    'Dirichlet',
    'DirichletMultinomial',
    'Distribution',
    'DoublesidedMaxwell',
    'Empirical',
    'ExpGamma',
    'ExpInverseGamma',
    'Exponential',
    'ExponentiallyModifiedGaussian',
    'ExpRelaxedOneHotCategorical',
    'FiniteDiscrete',
    'FULLY_REPARAMETERIZED',
    'Gamma',
    'GammaGamma',
    'GaussianProcess',
    'GaussianProcessRegressionModel',
    'GeneralizedExtremeValue',
    'GeneralizedNormal',
    'GeneralizedPareto',
    'Geometric',
    'Gumbel',
    'HalfCauchy',
    'HalfNormal',
    'HalfStudentT',
    'HiddenMarkovModel',
    'Horseshoe',
    'Independent',
    'independent_joint_distribution_from_structure',
    'InverseGamma',
    'InverseGaussian',
    'JohnsonSU',
    'JointDistribution',
    'JointDistributionCoroutine',
    'JointDistributionCoroutineAutoBatched',
    'JointDistributionNamed',
    'JointDistributionNamedAutoBatched',
    'JointDistributionSequential',
    'JointDistributionSequentialAutoBatched',
    'kl_divergence',
    'Kumaraswamy',
    'LambertWDistribution',
    'LambertWNormal',
    'Laplace',
    'LinearGaussianStateSpaceModel',
    'LKJ',
    'Logistic',
    'LogitNormal',
    'LogLogistic',
    'LogNormal',
    'MarkovChain',
    'Masked',
    'MatrixNormalLinearOperator',
    'MatrixTLinearOperator',
    # 'Mixture',
    'MixtureSameFamily',
    'Moyal',
    'Multinomial',
    'MultivariateNormalDiag',
    'MultivariateNormalDiagPlusLowRank',
    'MultivariateNormalFullCovariance',
    'MultivariateNormalLinearOperator',
    'MultivariateNormalTriL',
    'MultivariateStudentTLinearOperator',
    'mvn_conjugate_linear_update',
    'NegativeBinomial',
    'Normal',
    'normal_conjugates_known_scale_posterior',
    'normal_conjugates_known_scale_predictive',
    'NormalInverseGaussian',
    'NOT_REPARAMETERIZED',
    'OneHotCategorical',
    'OrderedLogistic',
    'Pareto',
    'PERT',
    # 'PixelCNN',
    'PlackettLuce',
    'Poisson',
    'PoissonLogNormalQuadratureCompound',
    'PowerSpherical',
    'ProbitBernoulli',
    'quadrature_scheme_lognormal_gauss_hermite',
    'quadrature_scheme_lognormal_quantiles',
    'QuantizedDistribution',
    'RegisterKL',
    'RelaxedBernoulli',
    'RelaxedOneHotCategorical',
    'ReparameterizationType',
    'Sample',
    'SigmoidBeta',
    'SinhArcsinh',
    'Skellam',
    'SphericalUniform',
    'StoppingRatioLogistic',
    'StudentT',
    'StudentTProcess',
    'TransformedDistribution',
    'Triangular',
    'TruncatedCauchy',
    'TruncatedNormal',
    'Uniform',
    'VariationalGaussianProcess',
    'VectorDeterministic',
    'VectorExponentialDiag',
    'VonMises',
    'VonMisesFisher',
    'Weibull',
    'WishartLinearOperator',
    'WishartTriL',
    'Zipf',
]


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# This file is auto-generated by substrates/meta/rewrite.py
# It will be surfaced by the build system as a symlink at:
#   `tensorflow_probability/substrates/numpy/distributions/__init__.py`
# For more info, see substrate_runfiles_symlinks in build_defs.bzl
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
