from setuptools import setup, find_packages
from mudpy import __version__,__license__,__author__

setup(
        name             = 'MudPy',
        version          = __version__,
        description      = 'music analysis tools for recommendation with deep learning',
        license          = __license__,
        author           = __author__,
        author_email     = 'narittan@gmail.com',
        url              = 'https://github.com/RittaNarita/MudPy',
        keywords         = 'MUsic Deep learning',
        packages         = ['mudpy'],
        install_requires = [],
        )