from setuptools import setup, find_packages

setup(
    name='mydllib',
    version='0.0.5',
    author="Jesus Castillo",
    author_email="jescas981@gmail.com",
    packages=find_packages(),
    install_requires=['torch','numpy','matplotlib','scikit-learn']
)