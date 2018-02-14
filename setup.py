#!/usr/bin/env python3

import ast
from os import path
import re
from setuptools import find_packages, setup

# read version from python module:
with open(path.join(path.dirname(__file__), 'smart_importer', '__init__.py'), 'rb') as f:
    file_contents = f.read().decode('utf-8')
    VERSION = str(ast.literal_eval(re.search(r'__version__\s+=\s+(.*)', file_contents).group(1)))
    COPYRIGHT = str(ast.literal_eval(re.search(r'__copyright__\s+=\s+(.*)', file_contents).group(1)))
    LICENSE = str(ast.literal_eval(re.search(r'__license__\s+=\s+(.*)', file_contents).group(1)))


# read version from python module:


# read readme.rst:
with open(path.join(path.dirname(__file__), 'README.rst')) as readme:
    LONG_DESCRIPTION = readme.read()


setup(
    name='smart_importer',
    version=VERSION,
    description='Smart importer for beancount and fava.',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/johannesjh/smart_importer',
    author='Johannes Harms',
    license=LICENSE,
    keywords='fava beancount accounting import csv machinelearning scikit-learn sklearn',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=[
        'beancount>=2.0b15',
        'scikit-learn>=0.19',
        'numpy>= 1.8.2',
        'scipy>=0.13.3'
    ],
    zip_safe=False,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Web Environment',
        'Environment :: Console',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Office/Business :: Financial :: Accounting',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
)