# pylint: disable=all

from setuptools import setup, find_packages


setup(
    name="nist_lookup",
    version="v2.11",
    packages=find_packages(exclude='test'),
    scripts=[
    ],

    install_requires=[
        'sqlalchemy',
        'numpy',
        'scipy',
        'beautifulsoup4',
    ],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.db'],
    },

    # metadata for upload to PyPI
    author="TOMCAT DPC group",
    author_email="",
    description="Download table from the NIST database",
    license="GNU GPL 3",
    keywords="",
    # project home page, if any
    url="https://git.psi.ch/tomcat/nist_lookup",
    # could also include long_description, download_url, classifiers, etc.
)
