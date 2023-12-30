from setuptools import setup, find_packages

setup(
    name='cerberus_ts',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pandas",
        "accelerate",
        "matplotlib"
    ],
)
