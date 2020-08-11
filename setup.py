from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name="moleculepred",
    version="0.0.1",
    author="Ehsan Zahedi",
    description="Deep learning classifiers for molecular property prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires='>=3.6',
    install_requires=[
            'tensorflow',
            'numpy',
            'pandas',
            'scikit-learn',
            'scipy',
        ],
)