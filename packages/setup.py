#!/usr/bin/env python
import os
from setuptools import setup, find_packages


if os.path.exists('README.md'):
    # Get the long description from the README file
    try:
        with open('README.md', encoding='utf-8') as f:
            long_description = f.read()
    except:
        long_description = ''
else:
    long_description = ''


setup(
    name='deepracer_gym',
    version='0.0.1',
    author='Uzair Akbar',
    author_email='uzair.akbar@gatech.edu',
    description='A gym environment for AWS DeepRacer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.gatech.edu/rldm/P4_deepracer',
    install_requires=[
        'numpy',
        'matplotlib',
        'gymnasium',
        'pyzmq',
        'msgpack',
        'msgpack_numpy',
        'loguru'
    ],
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
