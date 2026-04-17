#include "GpuDeviceRegistry.h"
#include <QDebug>

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

void GpuDeviceRegistry::enumerate() {
#ifdef HAVE_OPENCL
    m_devices.clear();
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // First pass: collect GPU devices from all platforms
        for (auto& p : platforms) {
            std::string pname;
            try { pname = p.getInfo<CL_PLATFORM_NAME>(); } catch (...) {} // GCOVR_EXCL_LINE

            std::vector<cl::Device> gpuDevs;
            try { p.getDevices(CL_DEVICE_TYPE_GPU, &gpuDevs); } catch (...) {} // GCOVR_EXCL_LINE

            for (auto& d : gpuDevs) {
                GpuDeviceInfo info;
                try {
                    info.name = QString::fromStdString(d.getInfo<CL_DEVICE_NAME>()).trimmed();
                } catch (...) { info.name = "Unknown GPU"; } // GCOVR_EXCL_LINE
                info.platformName = QString::fromStdString(pname).trimmed();
                m_devices.push_back(info);
                qDebug() << "[GpuRegistry] Found:" << info.name << "on" << info.platformName;
            }
        }

        // GCOVR_EXCL_START
        // Fallback: if no GPU devices found, include everything.
        // Infeasible on any machine with a GPU (test env requires one).
        if (m_devices.empty()) {
            for (auto& p : platforms) {
                std::string pname;
                try { pname = p.getInfo<CL_PLATFORM_NAME>(); } catch (...) {}
                std::vector<cl::Device> allDevs;
                try { p.getDevices(CL_DEVICE_TYPE_ALL, &allDevs); } catch (...) {}
                for (auto& d : allDevs) {
                    GpuDeviceInfo info;
                    try {
                        info.name = QString::fromStdString(d.getInfo<CL_DEVICE_NAME>()).trimmed();
                    } catch (...) { info.name = "Unknown Device"; }
                    info.platformName = QString::fromStdString(pname).trimmed();
                    m_devices.push_back(info);
                }
            }
        }
        // GCOVR_EXCL_STOP
    }
    // GCOVR_EXCL_START
    catch (...) {
        qWarning() << "[GpuRegistry] OpenCL enumeration failed";
    }
    // GCOVR_EXCL_STOP

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

// Enumerate all GPU devices in the same order as GpuDeviceRegistry::enumerate().
static std::vector<std::pair<cl::Platform, cl::Device>> enumerateAllOCLDevices() {
    std::vector<std::pair<cl::Platform, cl::Device>> result;
    std::vector<cl::Platform> platforms;
    try { cl::Platform::get(&platforms); } catch (...) { return result; }

    for (auto& p : platforms) {
        std::vector<cl::Device> gpuDevs;
        try { p.getDevices(CL_DEVICE_TYPE_GPU, &gpuDevs); } catch (...) {}
        for (auto& d : gpuDevs) result.push_back({p, d});
    }

    // GCOVR_EXCL_START
    if (result.empty()) {
        for (auto& p : platforms) {
            std::vector<cl::Device> allDevs;
            try { p.getDevices(CL_DEVICE_TYPE_ALL, &allDevs); } catch (...) {}
            for (auto& d : allDevs) result.push_back({p, d});
        }
    }
    // GCOVR_EXCL_STOP
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
