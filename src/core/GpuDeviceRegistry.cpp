#include "GpuDeviceRegistry.h"
#include <QDebug>
#include <algorithm>

#ifdef HAVE_OPENCL
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuDeviceRegistryOCL.h"
#endif

// ============================================================================
// GpuDeviceRegistry
// ============================================================================

GpuDeviceRegistry& GpuDeviceRegistry::instance() {
    static GpuDeviceRegistry reg;
    return reg;
}

#ifdef HAVE_OPENCL
static QString deviceTypeName(cl_device_type t) {
    if (t & CL_DEVICE_TYPE_GPU)         return "GPU";
    if (t & CL_DEVICE_TYPE_CPU)         return "CPU";
    if (t & CL_DEVICE_TYPE_ACCELERATOR) return "Accelerator";
    return "Device"; // GCOVR_EXCL_LINE
}

// Sort key: GPU(0) < CPU(1) < Accelerator(2) < other(3) so GPUs stay first
// in the picker and remain the default selection.
static int deviceTypeRank(cl_device_type t) {
    if (t & CL_DEVICE_TYPE_GPU)         return 0;
    if (t & CL_DEVICE_TYPE_CPU)         return 1;
    if (t & CL_DEVICE_TYPE_ACCELERATOR) return 2;
    return 3; // GCOVR_EXCL_LINE
}
#endif

void GpuDeviceRegistry::enumerate() {
#ifdef HAVE_OPENCL
    m_devices.clear();

    struct Entry { GpuDeviceInfo info; int rank; };
    std::vector<Entry> entries;

    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for (auto& p : platforms) {
            std::string pname;
            try { pname = p.getInfo<CL_PLATFORM_NAME>(); } catch (...) {} // GCOVR_EXCL_LINE

            std::vector<cl::Device> devs;
            try { p.getDevices(CL_DEVICE_TYPE_ALL, &devs); } catch (...) {} // GCOVR_EXCL_LINE

            for (auto& d : devs) {
                cl_device_type t = 0;
                try { t = d.getInfo<CL_DEVICE_TYPE>(); } catch (...) {} // GCOVR_EXCL_LINE

                GpuDeviceInfo info;
                try {
                    info.name = QString::fromStdString(d.getInfo<CL_DEVICE_NAME>()).trimmed();
                } catch (...) { info.name = "Unknown Device"; } // GCOVR_EXCL_LINE
                info.platformName = QString::fromStdString(pname).trimmed();
                info.typeName     = deviceTypeName(t);
                entries.push_back({info, deviceTypeRank(t)});
            }
        }
    }
    // GCOVR_EXCL_START
    catch (...) {
        qWarning() << "[GpuRegistry] OpenCL enumeration failed";
    }
    // GCOVR_EXCL_STOP

    std::stable_sort(entries.begin(), entries.end(),
                     [](const Entry& a, const Entry& b) { return a.rank < b.rank; });
    m_devices.reserve(entries.size());
    for (auto& e : entries) {
        m_devices.push_back(e.info);
        qDebug() << "[GpuRegistry] Found:" << e.info.name
                 << "(" << e.info.typeName << ") on" << e.info.platformName;
    }

    if (m_currentIndex >= static_cast<int>(m_devices.size()))
        m_currentIndex = 0;

    qDebug() << "[GpuRegistry]" << m_devices.size() << "device(s) found, using index" << m_currentIndex;
#endif
}

void GpuDeviceRegistry::setDevice(int index) {
    if (index == m_currentIndex) return;
    m_currentIndex = index;
    m_revision.fetch_add(1, std::memory_order_relaxed);
    qDebug() << "[GpuRegistry] Switched to device" << index
             << (index < static_cast<int>(m_devices.size()) ? m_devices[index].name : "?");
}

// ============================================================================
// GpuDeviceRegistryOCL (OpenCL companion — only compiled when HAVE_OPENCL)
// ============================================================================

#ifdef HAVE_OPENCL

// Enumerate all OpenCL devices in the same order as GpuDeviceRegistry::enumerate():
// GPUs first, then CPUs, then accelerators.
static std::vector<std::pair<cl::Platform, cl::Device>> enumerateAllOCLDevices() {
    std::vector<std::pair<cl::Platform, cl::Device>> result;
    std::vector<cl::Platform> platforms;
    try { cl::Platform::get(&platforms); } catch (...) { return result; } // GCOVR_EXCL_LINE

    struct Entry { cl::Platform p; cl::Device d; int rank; };
    std::vector<Entry> entries;
    for (auto& p : platforms) {
        std::vector<cl::Device> devs;
        try { p.getDevices(CL_DEVICE_TYPE_ALL, &devs); } catch (...) {} // GCOVR_EXCL_LINE
        for (auto& d : devs) {
            cl_device_type t = 0;
            try { t = d.getInfo<CL_DEVICE_TYPE>(); } catch (...) {} // GCOVR_EXCL_LINE
            entries.push_back({p, d, deviceTypeRank(t)});
        }
    }
    std::stable_sort(entries.begin(), entries.end(),
                     [](const Entry& a, const Entry& b) { return a.rank < b.rank; });
    result.reserve(entries.size());
    for (auto& e : entries) result.push_back({e.p, e.d});
    return result;
}

namespace GpuDeviceRegistryOCL {

bool getSelectedDevice(cl::Device& outDevice, cl::Platform& outPlatform) {
    auto devs = enumerateAllOCLDevices();
    if (devs.empty()) return false;
    int idx = GpuDeviceRegistry::instance().currentIndex();
    if (idx < 0 || idx >= static_cast<int>(devs.size())) idx = 0;
    outDevice   = devs[idx].second;
    outPlatform = devs[idx].first;
    return true;
}

} // namespace GpuDeviceRegistryOCL

#endif // HAVE_OPENCL
