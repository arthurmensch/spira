import os.path

import os
from os.path import join

import numpy

from sklearn._build_utils import get_blas_info

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('impl', parent_package, top_path)

    cblas_libs, blas_info = get_blas_info()

    config.add_extension('dataset_fast',
                         sources=['dataset_fast.c'],
                         include_dirs=[numpy.get_include()])

    config.add_extension('matrix_fact_fast',
                         sources=['matrix_fact_fast.c'],
                         include_dirs=[numpy.get_include()])

    config.add_extension('preprocessing_fast',
                         sources=['preprocessing_fast.c'],
                         include_dirs=[numpy.get_include()])

    config.add_extension('dict_fact_fast',
                         sources=['dict_fact_fast.c'],
                         libraries=cblas_libs,
                         include_dirs=[join('..', 'src', 'cblas'),
                                       numpy.get_include(),
                                       blas_info.pop('include_dirs', [])],
                         extra_compile_args=blas_info.pop('extra_compile_args',
                                                          []), **blas_info)

    config.add_subpackage('tests')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
