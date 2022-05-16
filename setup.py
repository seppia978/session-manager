from setuptools import setup, find_packages

from SessionManager import __version__

setup(
    name='session-manager-978',
    version=__version__,

    url='https://github.com/seppia978/session-manager',
    author='seppis978',
    author_email='samuele.poppi@unimore.it',

    py_modules=find_packages(),
    packages=find_packages(),
)