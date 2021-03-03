from setuptools import find_packages, setup

# https://medium.com/analytics-vidhya/how-to-create-a-python-library-7d5aea80cc3f
# https://towardsdatascience.com/make-your-own-python-package-6d08a400fc2d

setup(
    name='junet',
    packages=find_packages(include=['mypythonlib']),
    version='0.0.1',
    description='library for AI Template',
    author='June',
    license='MIT',
    install_requires=[],
    setup_requires=['tensorflow', 'tensorflow_addons'],
    tests_require=['tensorflow', 'tensorflow_addons'],
    test_suite='test',
)
