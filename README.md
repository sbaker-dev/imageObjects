# imageObjects

This acts as an API of sorts for CV2, with some common methods cast into objects that work with images, contours and 
more. 

### ContourObject
open-cv python has contours, a extremely useful component that can be extracted from images. However, these components 
are in a ndarray format that can then be processed by a list of methods, rather than an object which has a list of 
methods. This library changes that around and allows for the contour extracted via open cv python to be placed in an 
object which can then be manipulated. 

### ImageObject
There are lots of common image functions that have been mapped to images, so that a loaded image can have these as 
properties or as methods inherently rather than having to code them yourself.
