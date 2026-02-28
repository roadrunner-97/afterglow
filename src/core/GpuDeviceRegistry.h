#ifndef GPUDEVICEREGISTRY_H
#define GPUDEVICEREGISTRY_H

#include <QString>
#include <QStringList>
#include <atomic>
#include <vector>

struct GpuDeviceInfo {
    QString name;
    QString platformName;
};

/**
 * @brief Singleton registry of available OpenCL GPU devices.
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

    int currentIndex() const { return m_currentIndex; }

    // Starts at 1; effects initialise their stored revision to 0 so the very
    // first call always triggers init(). Bumped again on each setDevice() call.
    int revision() const { return m_revision.load(std::memory_order_relaxed); }

private:
    GpuDeviceRegistry() = default;

    std::vector<GpuDeviceInfo> m_devices;
    int m_currentIndex = 0;
    std::atomic<int> m_revision{1};
};

#endif // GPUDEVICEREGISTRY_H
