import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="metagenome-nmf",
    version="0.0.1",
    description="Model selection and inspection tools to apply NMF to metagenomic data",
    long_description=README,
    long_description_content_type="text/markdown",
    url="TODO",
    author="Anthony Duncan",
    author_email="a.duncan@uea.ac.uk",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["mg_nmf", "mg_nmf.nmf", "mg_nmf.gui", "mg_nmf.longhurst"],
    include_package_data=True,
    # install_requires=["feedparser", "html2text"],
    entry_points={
        "console_scripts": [
            "nmf=mg_nmf.nmf.__main__:main",
        ]
    },
)
