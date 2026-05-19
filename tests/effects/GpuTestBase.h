#pragma once

#include <QObject>
#include <QTest>
#include "GpuDeviceRegistry.h"

// Base class for effect unit tests that exercise the GPU pipeline.
// Provides initTestCase() (runs once before all test methods) that enumerates
// OpenCL devices and skips the entire suite if none are found.
// Subclasses inherit m_hasGpu and guard per-test GPU calls with:
//     if (!m_hasGpu) QSKIP("No GPU");
class GpuTestBase : public QObject {
    Q_OBJECT
protected:
    bool m_hasGpu = false;

protected slots:
    void initTestCase() {
        GpuDeviceRegistry::instance().enumerate();
        if (GpuDeviceRegistry::instance().count() == 0) QSKIP("No OpenCL device found — skipping GPU effect tests");
        GpuDeviceRegistry::instance().setDevice(0);
        m_hasGpu = true;
    }
};
