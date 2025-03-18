# Spatial resampling Using Descriptors Manual

This manual is intended to facilitate the understanding of spatial resampling using Descriptors.

### Using Pre-Generated Spatial Descriptors

To execute a Spatial resampling process generating and using a spatial descriptor, the value of `--SpatialDescriptorMode` should be set to `GeneratingDescriptor` and `UsingDescriptor`, respectively. The spatial resampling is conducted without generating and using the spatial descriptor if the value of `--SpatialDescriptorMode` is set to `NoDescriptor`. You can set the name of the spatial descriptor with a path configured by `--SpatialDescriptor`. The spatial descriptor can be used for cross-checking.

`--SpatialDescriptorMode`
  - `GeneratingDescriptor`: save the spatial descriptor to the file 
  - `GeneratingDescriptorExit`: save the spatial descriptor to the fileand exit.
  - `UsingDescriptor`: run the spatial resampling using the spatial descriptor from the file.
  - `NoDescriptor (default)`: run the spatial resampling without generating and using the spatial descriptor (from a file).

`--SpatialDescriptor` : File name of spatial descriptor file to be saved and loaded.

### Default Behavior
The default of `--SpatialDescriptorMode` is `NoDescriptor`. Therefore, VCM-RS is conducted to run the spatial resampling without generating and using the spatial descriptor.

