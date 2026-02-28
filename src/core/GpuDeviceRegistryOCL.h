#ifndef GPUDEVICEREGISTRYOCL_H
#define GPUDEVICEREGISTRYOCL_H

// Companion header for OpenCL-using translation units.
// Include this AFTER opencl.hpp has already been included (so cl::Device and
// cl::Platform are defined). It declares GpuDeviceRegistryOCL::getSelectedDevice()
// which is implemented in GpuDeviceRegistry.cpp.

#ifdef HAVE_OPENCL
#include "GpuDeviceRegistry.h"

namespace GpuDeviceRegistryOCL {

// Returns the cl::Device and cl::Platform chosen by GpuDeviceRegistry.
// Returns false if no OpenCL device is available at all.
bool getSelectedDevice(cl::Device& outDevice, cl::Platform& outPlatform);

} // namespace GpuDeviceRegistryOCL
#endif // HAVE_OPENCL

#endif // GPUDEVICEREGISTRYOCL_H
