from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'numpy',
    'scipy',
    'h5py',
    'nibabel',
    'tqdm',
    'pandas',
]

try:
    import tensorflow
except ModuleNotFoundError:
    REQUIRED_PACKAGES.append('tensorflow')

setup(
    name='JuneNet',
    version='0.8',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    requires=[]
)
