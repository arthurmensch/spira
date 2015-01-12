import os.path

import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('impl', parent_package, top_path)

    config.add_extension('matrix_fact_fast',
                         sources=['matrix_fact_fast.c'],
                         include_dirs=[numpy.get_include()])

    config.add_extension('preprocessing_fast',
                         sources=['preprocessing_fast.c'],
                         include_dirs=[numpy.get_include()])

    config.add_subpackage('tests')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
