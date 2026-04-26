#ifndef GPUDEVICEREGISTRY_H
#define GPUDEVICEREGISTRY_H

#include <QString>
#include <QStringList>
#include <atomic>
#include <vector>

struct GpuDeviceInfo {
    QString name;
    QString platformName;
    QString typeName;   // "GPU", "CPU", "Accelerator", or "Device"
};

/**
 * @brief Singleton registry of available OpenCL compute devices.
 *
 * Enumerates every device exposed by the OpenCL ICDs on the system: discrete
 * and integrated GPUs first, then CPU runtimes (POCL, Intel CPU, AMD CPU…),
 * then accelerators. A box with no GPU still gets a working device list as
 * long as a CPU ICD is installed.
 *
 * Call enumerate() once at startup. The UI reads deviceNames() to populate a
 * combo box and calls setDevice(idx) on selection. setDevice() bumps revision()
 * so every effect's GpuContext knows to reinitialise on the next GPU call.
 */
class GpuDeviceRegistry {
public:
    static GpuDeviceRegistry& instance();

    // Populate m_devices by querying OpenCL. No-op if OpenCL is unavailable.
    void enumerate();

    const std::vector<GpuDeviceInfo>& devices() const { return m_devices; }
    int count() const { return static_cast<int>(m_devices.size()); }

    // Switch the active device. Bumps revision() so GpuContexts reinitialise.
    void setDevice(int index);

    int currentIndex() const { return m_currentIndex.load(std::memory_order_relaxed); }

    // Starts at 1; effects initialise their stored revision to 0 so the very
    // first call always triggers init(). Bumped again on each setDevice() call.
    int revision() const { return m_revision.load(std::memory_order_relaxed); }

private:
    GpuDeviceRegistry() = default;

    std::vector<GpuDeviceInfo> m_devices;
    std::atomic<int> m_currentIndex{0};
    std::atomic<int> m_revision{1};
};

#endif // GPUDEVICEREGISTRY_H
