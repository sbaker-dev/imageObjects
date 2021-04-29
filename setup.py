# Copyright (C) 2020 Samuel Baker

DESCRIPTION = "Helper Objects for working with open computer vision and image projects"
LONG_DESCRIPTION = """
# imageObjects

This acts as an API of sorts for CV2, with some common methods cast into objects that work with images, contours and 
more. All source code at the [github repo page][repo]

### ContourObject
open-cv python has contours, a extremely useful component that can be extracted from images. However, these components 
are in a ndarray format that can then be processed by a list of methods, rather than an object which has a list of 
methods. This library changes that around and allows for the contour extracted via open cv python to be placed in an 
object which can then be manipulated. 

### ImageObject
There are lots of common image functions that have been mapped to images, so that a loaded image can have these as 
properties or as methods inherently rather than having to code them yourself.

### LineObject
This will search for lines within an ImageObject which you can then utilise as a mask or manipulate with ImageObjects

### ImageMaker
This can be used to make new ImageObjects, when you don't have anything to load or custom ImageObjects like text-boxes

### common
Whilst common is mostly designed for internal use, there are some common methods here that may save you re-defining them
yourself or from other packages


[repo]: https://github.com/sbaker-dev/imageObjects
"""

LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

DISTNAME = 'imageObjects'
MAINTAINER = 'Samuel Baker'
MAINTAINER_EMAIL = 'samuelbaker.researcher@gmail.com'
LICENSE = 'MIT'
DOWNLOAD_URL = "https://github.com/sbaker-dev/contourObject"
VERSION = "0.12.1"
PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    'opencv-python',
    'numpy',
    'matplotlib',
    'vectorObjects',
    'shapely',
    'miscSupports'
]

CLASSIFIERS = [
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'License :: OSI Approved :: MIT License',
]

if __name__ == "__main__":

    from setuptools import setup, find_packages

    import sys

    if sys.version_info[:2] < (3, 7):
        raise RuntimeError("imageObjects requires python >= 3.7.")

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
        packages=find_packages(),
        classifiers=CLASSIFIERS
    )
