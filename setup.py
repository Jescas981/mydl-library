from setuptools import setup, find_packages

setup(
    name='dllib',
    version='0.0.1',
    author="Jesus Castillo",
    author_email="jescas981@gmail.com",
    packages=find_packages(),
    install_requires=['torch','numpy','matplotlib','scikit-learn']
)