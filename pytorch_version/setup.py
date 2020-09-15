#nsml: nsml/default_ml:cuda9_torch1.0
from distutils.core import setup
import setuptools

setup(
    name='ncc_test',
    version='1.0',
    install_requires=['scikit-learn' ,'matplotlib','opencv-python']
)