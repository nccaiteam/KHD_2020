#nsml: nvcr.io/nvidia/tensorflow:20.06-tf2-py3
from distutils.core import setup
import setuptools

setup(
    name='ncc_test',
    version='1.0',
    install_requires=['scikit-learn' ,'matplotlib']
)