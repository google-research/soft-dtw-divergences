# Copyright 2020 Google LLC
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

"""Install sdtw_div."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='sdtw_div',
    version='0.1',
    description=(
        'Differentiable Divergences between Time Series.'),
    author='Google LLC',
    author_email='no-reply@google.com',
    url='https://github.com/google-research/soft-dtw-divergences',
    license='Apache2',
    packages=find_packages(),
    package_data={},
    install_requires=[
        'numba',
        'numpy',
        'scipy>=1.2.0',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='machine learning time series dtw',
)
