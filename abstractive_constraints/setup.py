import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1'
PACKAGE_NAME = 'abstractive_constraints'
AUTHOR = 'Markus Dreyer'
AUTHOR_EMAIL = 'mddreyer@amazon.com'
URL = 'https://github.com/markusdr'

LICENSE = 'MIT No Attribution'
DESCRIPTION = 'Abstractiveness constraints for decoding'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = ["more_itertools"
    ]

setup(name=PACKAGE_NAME,
            version=VERSION,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            long_description_content_type=LONG_DESC_TYPE,
            author=AUTHOR,
            license=LICENSE,
            author_email=AUTHOR_EMAIL,
            url=URL,
            install_requires=INSTALL_REQUIRES,
            packages=find_packages()
            )
