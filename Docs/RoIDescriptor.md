# RoI Based Coding Using Descriptors Manual

This manual is intended to facilitate the understanding of RoI based coding using Descriptors.

### Using Pre-Generated RoI Descriptors

To execute an RoI based process using a pre-generated RoI descriptor, the value of `--RoIDescriptor` should be set to the descriptor file. When descriptor usage mode is activated, the object detection part from the video is omitted, and the RoI based process is carried out by reading the descriptor. This can be used for cross-checking using descriptors provided by the proponent.

### Default Behavior

If `--RoIDescriptor` is None, the RoI will be extracted from the video as usual, and the RoI based process will be executed.

`--RoIDescriptorMode`
  - `save`: save the spatial descriptor to the file 
  - `saveexit`: save the spatial descriptor to the fileand exit.
  - `load`: run the spatial resampling using the spatial descriptor from the file.
  - `generate`: run the spatial resampling without generating and using the spatial descriptor (from a file).
