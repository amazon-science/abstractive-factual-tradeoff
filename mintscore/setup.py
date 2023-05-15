import os
import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1'
PACKAGE_NAME = 'mintscore'
AUTHOR = 'Markus Dreyer'
AUTHOR_EMAIL = 'mddreyer@amazon.com'
URL = 'https://github.com/markusdr'

LICENSE = 'MIT No Attribution'
DESCRIPTION = 'MINT score to measure abstractiveness'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

PATH=os.path.dirname(os.path.realpath(__file__))
MINTLCS_PATH=os.path.realpath(f'{PATH}/../mintlcs')

INSTALL_REQUIRES = [
    "spacy",
    "scipy",
    "joblib",
    f"mintlcs @ file://localhost/{MINTLCS_PATH}"
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
      packages=find_packages(),
      entry_points={
          "console_scripts": [
              "mint = mintscore.mint:run"
          ]
      },
)
