"""Specifies pip installation details"""

from setuptools import find_packages, setup

exec(open('src/_version.py').read())

DESCRIPTION_STR = '''
Workflows for supervised classification of the spatial distribution of
Norwegian Vegetation Types.
'''

setup(
    name='dmvtnor',
    packages=find_packages(include=['src', 'src.*']),
    version=__version__,
    description=DESCRIPTION_STR,
    author='Lasse Keetz',
    keywords=[
        "Distribution modelling", "Machine learning",
        "Supervised classification", "Remote sensing",
        "Land cover", "Vegetation mapping"
    ],
    cmdclass={'bdist_wheel': None},
)
