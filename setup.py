# Copyright (C) 2020 Samuel Baker

DESCRIPTION = "Open computer vision contours as objects"
LONG_DESCRIPTION = """
# contourObject

open-cv python has contours, a extremely useful component that can be extracted from images. However, these components 
are in a ndarray format that can then be processed by a list of methods, rather than an object which has a list of 
methods. This library changes that around and allows for the contour extracted via open cv python to be placed in an 
object which can then be manipulated. 

"""
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

DISTNAME = 'contourObject'
MAINTAINER = 'Samuel Baker'
MAINTAINER_EMAIL = 'samuelbaker.researcher@gmail.com'
LICENSE = 'MIT'
DOWNLOAD_URL = "https://github.com/sbaker-dev/contourObject"
VERSION = "0.02.0"
PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    'opencv-python',
    'numpy']

PACKAGES = [
    "contourObject",
]

CLASSIFIERS = [
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'License :: OSI Approved :: MIT License',
]

if __name__ == "__main__":

    from setuptools import setup

    import sys

    if sys.version_info[:2] < (3, 7):
        raise RuntimeError("contourObject requires python >= 3.7.")

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        license=LICENSE,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
        classifiers=CLASSIFIERS
    )
