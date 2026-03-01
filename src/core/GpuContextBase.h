#ifndef GPUCONTEXTBASE_H
#define GPUCONTEXTBASE_H

// CRTP base for per-effect GPU contexts.
//
// Include this AFTER opencl.hpp and GpuDeviceRegistryOCL.h:
//
//   #define CL_HPP_TARGET_OPENCL_VERSION 200
//   ...
//   #include <CL/opencl.hpp>
//   #include "GpuDeviceRegistryOCL.h"
//   #include "GpuContextBase.h"
//
// Usage inside the effect's anonymous namespace:
//
//   struct GpuContext : GpuContextBase<GpuContext> {
//       cl::Kernel kernelFoo;
//       cl::Kernel kernelBar;
//       void init() {
//           cl::Device dev;
//           if (!acquireDevice(dev, "MyEffect")) return;
//           try {
//               cl::Program prog(context, GPU_KERNEL_SOURCE);
//               prog.build({dev});
//               kernelFoo = cl::Kernel(prog, "myKernel");
//               available = true;
//               qDebug() << "[GPU] MyEffect ready on:"
//                        << QString::fromStdString(dev.getInfo<CL_DEVICE_NAME>());
//           } catch (const cl::Error& e) {
//               qWarning() << "[GPU] MyEffect init failed:" << e.what() << "(err" << e.err() << ")";
//           }
//       }
//   };
//
// All effects must call GpuContext::instance() with gpuMutex held.

#ifdef HAVE_OPENCL

#include "GpuDeviceRegistry.h"
#include "GpuDeviceRegistryOCL.h"
#include <QDebug>

template<typename Derived>
struct GpuContextBase {
    cl::Context      context;
    cl::CommandQueue queue;
    bool             available = false;
    int              m_revision = 0;

    // Must be called with the effect's gpuMutex held.
    // Handles device-change detection via GpuDeviceRegistry revision counter.
    static Derived& instance() {
        static Derived ctx;
        int rev = GpuDeviceRegistry::instance().revision();
        if (ctx.m_revision != rev) {
            ctx            = Derived{};
            ctx.m_revision = rev;
            ctx.init();
        }
        return ctx;
    }

protected:
    // Acquires the currently selected OpenCL device, then initialises
    // this->context and this->queue.  Returns true on success; logs a
    // warning (with the supplied effect name) and returns false on failure.
    bool acquireDevice(cl::Device& outDevice, const char* effectName) {
        cl::Platform platform;
        if (!GpuDeviceRegistryOCL::getSelectedDevice(outDevice, platform)) {
            qWarning() << "[GPU]" << effectName << ": no OpenCL device available";
            return false;
        }
        context = cl::Context(outDevice);
        queue   = cl::CommandQueue(context, outDevice);
        return true;
    }
};

#endif // HAVE_OPENCL
#endif // GPUCONTEXTBASE_H
