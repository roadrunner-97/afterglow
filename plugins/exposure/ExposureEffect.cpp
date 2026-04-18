#include "ExposureEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>
#include <QPainter>
#include <QPolygonF>
#include <QSizePolicy>
#include <cmath>
#include <mutex>

// ============================================================================
// GPU path (OpenCL)
// ============================================================================

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuDeviceRegistryOCL.h"
#include "GpuContextBase.h"

namespace {

// libraw outputs sRGB gamma-encoded data (output_color=1), so exposure must be
// applied in linear light: decode sRGB → scale by 2^EV → re-encode sRGB.
//
// Zone adjustments treat the four slider values as PCHIP control points anchored
// at the midpoint of each zone:
//   blacks=0.075  shadows=0.325  highlights=0.675  whites=0.925
// Zone widths: blacks=15%, shadows=35%, highlights=35%, whites=15%.
// The spline is evaluated at each pixel's luminance to get a smooth zone EV
// offset.  Pixels darker than 0.075 get exactly blacksEv; pixels brighter than
// 0.925 get exactly whitesEv.
//
// 8-bit path:  QImage::Format_RGB32  (uint = 0xFFRRGGBB), stride = bytesPerLine / 4
// 16-bit path: QImage::Format_RGBX64 (ushort4 per pixel), stride = bytesPerLine / 8
static const char* GPU_KERNEL_SOURCE = R"CL(
float srgb_to_linear(float v) {
    return v <= 0.04045f ? v * (1.0f / 12.92f)
                         : native_powr((v + 0.055f) * (1.0f / 1.055f), 2.4f);
}

float linear_to_srgb(float v) {
    v = clamp(v, 0.0f, 1.0f);
    return v <= 0.0031308f ? v * 12.92f
                           : 1.055f * native_powr(v, 1.0f / 2.4f) - 0.055f;
}

// PCHIP interpolation through control points at the midpoint of each zone:
//   blacks=0.075  shadows=0.325  highlights=0.675  whites=0.925
// Zone widths: blacks=15%, shadows=35%, highlights=35%, whites=15%.
// Segment lengths (midpoint-to-midpoint): h0=0.25, h1=0.35, h2=0.25.
// Cubic Hermite with zero endpoint slopes; interior tangents are the
// h-weighted arithmetic mean of adjacent slopes, zeroed on sign-change.
float zoneEv(float lum, float p0, float p1, float p2, float p3) {
    if (lum <= 0.075f) return p0;
    if (lum >= 0.925f) return p3;

    const float h0 = 0.25f, h1 = 0.35f, h2 = 0.25f;
    float d0 = (p1 - p0) / h0;
    float d1 = (p2 - p1) / h1;
    float d2 = (p3 - p2) / h2;

    float m1 = (d0 * d1 > 0.0f) ? (h0 * d1 + h1 * d0) / (h0 + h1) : 0.0f;
    float m2 = (d1 * d2 > 0.0f) ? (h1 * d2 + h2 * d1) / (h1 + h2) : 0.0f;

    float pk, pk1, mk, mk1, h, s;
    if (lum < 0.325f) {
        s = (lum - 0.075f) / h0;
        pk = p0; pk1 = p1; mk = 0.0f; mk1 = m1; h = h0;
    } else if (lum < 0.675f) {
        s = (lum - 0.325f) / h1;
        pk = p1; pk1 = p2; mk = m1;   mk1 = m2; h = h1;
    } else {
        s = (lum - 0.675f) / h2;
        pk = p2; pk1 = p3; mk = m2;   mk1 = 0.0f; h = h2;
    }

    float s2 = s * s, s3 = s2 * s;
    return (2.0f*s3 - 3.0f*s2 + 1.0f) * pk
         + (      s3 - 2.0f*s2 + s)   * mk * h
         + (-2.0f*s3 + 3.0f*s2)       * pk1
         + (      s3 - s2)             * mk1 * h;
}

__kernel void adjustExposure(__global uint* pixels,
                              int   stride,
                              int   width,
                              int   height,
                              float globalEv,
                              float blacksEv,
                              float shadowsEv,
                              float highlightsEv,
                              float whitesEv)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    uint pixel = pixels[y * stride + x];
    float r_s = ((pixel >> 16) & 0xFFu) / 255.0f;
    float g_s = ((pixel >>  8) & 0xFFu) / 255.0f;
    float b_s = ( pixel        & 0xFFu) / 255.0f;

    float lum = 0.2126f * r_s + 0.7152f * g_s + 0.0722f * b_s;
    float ev      = globalEv + zoneEv(lum, blacksEv, shadowsEv, highlightsEv, whitesEv);
    float evFactor = native_exp2(ev);

    float r = linear_to_srgb(srgb_to_linear(r_s) * evFactor);
    float g = linear_to_srgb(srgb_to_linear(g_s) * evFactor);
    float b = linear_to_srgb(srgb_to_linear(b_s) * evFactor);

    uint ri = (uint)(r * 255.0f + 0.5f);
    uint gi = (uint)(g * 255.0f + 0.5f);
    uint bi = (uint)(b * 255.0f + 0.5f);
    pixels[y * stride + x] = 0xFF000000u | (ri << 16) | (gi << 8) | bi;
}

__kernel void adjustExposure16(__global ushort4* pixels,
                                int   stride,
                                int   width,
                                int   height,
                                float globalEv,
                                float blacksEv,
                                float shadowsEv,
                                float highlightsEv,
                                float whitesEv)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    // Same structure as the 8-bit kernel: r_s/g_s/b_s are sRGB-encoded values;
    // lum is perceptual (sRGB) luminance used for zone detection; EV adjustment
    // is applied in linear light (srgb_to_linear → scale → linear_to_srgb).
    ushort4 px = pixels[y * stride + x];
    float r_s = px.s0 / 65535.0f;
    float g_s = px.s1 / 65535.0f;
    float b_s = px.s2 / 65535.0f;

    float lum = 0.2126f * r_s + 0.7152f * g_s + 0.0722f * b_s;
    float ev      = globalEv + zoneEv(lum, blacksEv, shadowsEv, highlightsEv, whitesEv);
    float evFactor = native_exp2(ev);

    px.s0 = (ushort)(clamp(linear_to_srgb(srgb_to_linear(r_s) * evFactor) * 65535.0f + 0.5f, 0.0f, 65535.0f));
    px.s1 = (ushort)(clamp(linear_to_srgb(srgb_to_linear(g_s) * evFactor) * 65535.0f + 0.5f, 0.0f, 65535.0f));
    px.s2 = (ushort)(clamp(linear_to_srgb(srgb_to_linear(b_s) * evFactor) * 65535.0f + 0.5f, 0.0f, 65535.0f));
    pixels[y * stride + x] = px;
}
)CL";

struct GpuContext : GpuContextBase<GpuContext> {
    cl::Kernel kernel;
    cl::Kernel kernel16;

    void init() {
        cl::Device device;
        if (!acquireDevice(device, "Exposure")) return;
        try {
            cl::Program prog(context, GPU_KERNEL_SOURCE);
            prog.build({device});
            kernel   = cl::Kernel(prog, "adjustExposure");
            kernel16 = cl::Kernel(prog, "adjustExposure16");
            available = true;
            qDebug() << "[GPU] Exposure ready on:"
                     << QString::fromStdString(device.getInfo<CL_DEVICE_NAME>());
        }
        // GCOVR_EXCL_START
        catch (const cl::Error& e) {
            qWarning() << "[GPU] Exposure init failed:" << e.what() << "(err" << e.err() << ")";
        }
        // GCOVR_EXCL_STOP
    }
};

static std::mutex gpuMutex;

struct ZoneEvs {
    float global, blacks, shadows, highlights, whites;
};

static void setKernelZoneArgs(cl::Kernel& k, cl::Buffer& buf, int stride, int w, int h,
                               const ZoneEvs& z) {
    k.setArg(0, buf);
    k.setArg(1, stride);
    k.setArg(2, w);
    k.setArg(3, h);
    k.setArg(4, z.global);
    k.setArg(5, z.blacks);
    k.setArg(6, z.shadows);
    k.setArg(7, z.highlights);
    k.setArg(8, z.whites);
}

static QImage processImageGPU(const QImage& image, const ZoneEvs& z) {
    QImage result = image.convertToFormat(QImage::Format_RGB32);
    const int    width    = result.width();
    const int    height   = result.height();
    const int    stride   = result.bytesPerLine() / 4;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer buf(gpu.context,
                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       bufBytes, result.bits());

        setKernelZoneArgs(gpu.kernel, buf, stride, width, height, z);
        gpu.queue.enqueueNDRangeKernel(gpu.kernel, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Exposure kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

static QImage processImageGPU16(const QImage& image, const ZoneEvs& z) {
    QImage result = image;
    const int    width    = result.width();
    const int    height   = result.height();
    const int    stride   = result.bytesPerLine() / 8;
    const size_t bufBytes = static_cast<size_t>(result.bytesPerLine()) * height;

    try {
        std::lock_guard<std::mutex> lock(gpuMutex);
        GpuContext& gpu = GpuContext::instance();
        if (!gpu.available) return {};

        cl::Buffer buf(gpu.context,
                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       bufBytes, result.bits());

        setKernelZoneArgs(gpu.kernel16, buf, stride, width, height, z);
        gpu.queue.enqueueNDRangeKernel(gpu.kernel16, cl::NullRange,
                                       cl::NDRange(width, height), cl::NullRange);
        gpu.queue.finish();
        gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, bufBytes, result.bits());
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GPU] Exposure16 kernel failed:" << e.what() << "(err" << e.err() << ")";
        return {};
    }
    // GCOVR_EXCL_STOP
    return result;
}

// ============================================================================
// ToneCurveWidget — CPU mirror of the GPU zone math, drawn as a graph
// ============================================================================

// CPU mirror of the GPU kernel — must stay in sync with the OpenCL source above.
static float cpuSrgbToLinear(float v) {
    return v <= 0.04045f ? v / 12.92f
                         : std::pow((v + 0.055f) / 1.055f, 2.4f);
}
static float cpuLinearToSrgb(float v) {
    v = std::max(0.0f, std::min(1.0f, v));
    return v <= 0.0031308f ? v * 12.92f
                           : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}
// CPU mirror of zoneEv() — must stay in sync with the OpenCL source above.
static float cpuZoneEv(float L, const ZoneEvs& z) {
    if (L <= 0.075f) return z.blacks;
    if (L >= 0.925f) return z.whites;

    constexpr float h0 = 0.25f, h1 = 0.35f, h2 = 0.25f;
    float d0 = (z.shadows    - z.blacks)     / h0;
    float d1 = (z.highlights - z.shadows)    / h1;
    float d2 = (z.whites     - z.highlights) / h2;

    float m1 = (d0 * d1 > 0.0f) ? (h0 * d1 + h1 * d0) / (h0 + h1) : 0.0f;
    float m2 = (d1 * d2 > 0.0f) ? (h1 * d2 + h2 * d1) / (h1 + h2) : 0.0f;

    float pk, pk1, mk, mk1, h, s;
    if (L < 0.325f) {
        s = (L - 0.075f) / h0;
        pk = z.blacks;     pk1 = z.shadows;    mk = 0.0f; mk1 = m1;   h = h0;
    } else if (L < 0.675f) {
        s = (L - 0.325f) / h1;
        pk = z.shadows;    pk1 = z.highlights; mk = m1;   mk1 = m2;   h = h1;
    } else {
        s = (L - 0.675f) / h2;
        pk = z.highlights; pk1 = z.whites;     mk = m2;   mk1 = 0.0f; h = h2;
    }

    float s2 = s*s, s3 = s2*s;
    return (2.0f*s3 - 3.0f*s2 + 1.0f) * pk
         + (      s3 - 2.0f*s2 + s)   * mk * h
         + (-2.0f*s3 + 3.0f*s2)       * pk1
         + (      s3 - s2)             * mk1 * h;
}
static float cpuCurve(float L, const ZoneEvs& z) {
    float ev = cpuZoneEv(L, z);   // zone adjustments only — global EV excluded from shape
    return cpuLinearToSrgb(cpuSrgbToLinear(L) * std::exp2(ev));
}

class ToneCurveWidget : public QWidget {
public:
    explicit ToneCurveWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        setFixedHeight(160);
    }

    void setParams(const ZoneEvs& z) { m_z = z; update(); }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        // Inner plot area with a small margin for the border
        const QRectF r = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5);
        const float W = r.width(), H = r.height();
        auto toScreen = [&](float L, float out) -> QPointF {
            return { r.left() + L * W, r.bottom() - out * H };
        };

        // Background gradient: black on left, white on right
        QLinearGradient bg(r.left(), 0, r.right(), 0);
        bg.setColorAt(0.0, QColor(20, 20, 20));
        bg.setColorAt(1.0, QColor(55, 55, 55));
        p.fillRect(r, bg);

        // Zone bands (subtle tinted backgrounds — palette matches app accent colours)
        // Widths: blacks=15%, shadows=35%, highlights=35%, whites=15%
        struct ZoneBand { float x0, x1; QColor col; };
        const ZoneBand bands[] = {
            { 0.00f, 0.15f, QColor( 91, 110, 168, 32) },  // blacks  — steel blue
            { 0.15f, 0.50f, QColor( 58, 136, 152, 26) },  // shadows — muted teal
            { 0.50f, 0.85f, QColor(192, 128,  44, 26) },  // highlights — warm amber
            { 0.85f, 1.00f, QColor(200, 168,  64, 32) },  // whites  — golden sand
        };
        for (const auto& b : bands) {
            QRectF br(r.left() + b.x0 * W, r.top(),
                      (b.x1 - b.x0) * W, H);
            p.fillRect(br, b.col);
        }

        // Vertical zone boundary lines at 0.15, 0.5, 0.85; horizontal at 0.25, 0.5, 0.75
        p.setPen(QPen(QColor(80, 80, 80, 140), 1));
        for (float frac : { 0.15f, 0.50f, 0.85f })
            p.drawLine(toScreen(frac, 0.0f), toScreen(frac, 1.0f));
        for (int i = 1; i <= 3; i++)
            p.drawLine(toScreen(0.0f, float(i) / 4.0f), toScreen(1.0f, float(i) / 4.0f));

        // Zone labels at the bottom of each band
        p.setPen(QColor(160, 160, 160, 160));
        QFont f = p.font();
        f.setPixelSize(9);
        p.setFont(f);
        const char* labels[] = { "Blacks", "Shadows", "Highlights", "Whites" };
        for (int i = 0; i < 4; i++) {
            float cx = (bands[i].x0 + bands[i].x1) / 2.0f;
            QPointF pt = toScreen(cx, 0.0f);
            QRectF lr(pt.x() - 30, pt.y() - 14, 60, 12);
            p.drawText(lr, Qt::AlignCenter, labels[i]);
        }

        // Identity diagonal (neutral / no adjustment)
        p.setPen(QPen(QColor(140, 140, 140, 100), 1, Qt::DashLine));
        p.drawLine(toScreen(0.0f, 0.0f), toScreen(1.0f, 1.0f));

        // Tone curve — sample at each pixel column for smooth result
        const int N = std::max(2, int(W));
        QPolygonF poly;
        poly.reserve(N + 1);
        for (int i = 0; i <= N; i++) {
            float L   = float(i) / float(N);
            float out = cpuCurve(L, m_z);
            poly << toScreen(L, out);
        }
        p.setPen(QPen(Qt::white, 1.5f));
        p.drawPolyline(poly);

        // Border
        p.setPen(QPen(QColor(90, 90, 90), 1));
        p.drawRect(r);

        // Base exposure label — shown when global EV is set, so it's not forgotten
        if (std::abs(m_z.global) >= 0.05f) {
            QString label = QString("Base: %1%2 EV")
                .arg(m_z.global > 0 ? "+" : "")
                .arg(double(m_z.global), 0, 'f', 1);
            p.setPen(QColor(200, 200, 200, 180));
            QFont lf = p.font();
            lf.setPixelSize(9);
            p.setFont(lf);
            QRectF lr(r.left() + 4, r.top() + 4, 80, 12);
            p.drawText(lr, Qt::AlignLeft | Qt::AlignVCenter, label);
        }
    }

private:
    ZoneEvs m_z{};
};

} // namespace

// ============================================================================
// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
// Exposure is applied in linear light (just rgb *= 2^ev); zone selection uses
// a perceptual L = linear_to_srgb(linear_luma(px)) so the zone midpoints stay
// where the UI expects them.  Do not clamp outputs — scene-linear values above
// 1.0 are valid HDR; the final pack kernel clamps once.
// ============================================================================
static const char* PIPELINE_KERNEL_SOURCE = COLOR_KERNELS_SRC R"CL(

// PCHIP interpolation through control points at the midpoint of each zone:
//   blacks=0.075  shadows=0.325  highlights=0.675  whites=0.925
// Zone widths: blacks=15%, shadows=35%, highlights=35%, whites=15%.
// Segment lengths (midpoint-to-midpoint): h0=0.25, h1=0.35, h2=0.25.
// Cubic Hermite with zero endpoint slopes; interior tangents are the
// h-weighted arithmetic mean of adjacent slopes, zeroed on sign-change.
float zoneEvLinear(float lum, float p0, float p1, float p2, float p3) {
    if (lum <= 0.075f) return p0;
    if (lum >= 0.925f) return p3;

    const float h0 = 0.25f, h1 = 0.35f, h2 = 0.25f;
    float d0 = (p1 - p0) / h0;
    float d1 = (p2 - p1) / h1;
    float d2 = (p3 - p2) / h2;

    float m1 = (d0 * d1 > 0.0f) ? (h0 * d1 + h1 * d0) / (h0 + h1) : 0.0f;
    float m2 = (d1 * d2 > 0.0f) ? (h1 * d2 + h2 * d1) / (h1 + h2) : 0.0f;

    float pk, pk1, mk, mk1, h, s;
    if (lum < 0.325f) {
        s = (lum - 0.075f) / h0;
        pk = p0; pk1 = p1; mk = 0.0f; mk1 = m1; h = h0;
    } else if (lum < 0.675f) {
        s = (lum - 0.325f) / h1;
        pk = p1; pk1 = p2; mk = m1;   mk1 = m2; h = h1;
    } else {
        s = (lum - 0.675f) / h2;
        pk = p2; pk1 = p3; mk = m2;   mk1 = 0.0f; h = h2;
    }

    float s2 = s * s, s3 = s2 * s;
    return (2.0f*s3 - 3.0f*s2 + 1.0f) * pk
         + (      s3 - 2.0f*s2 + s)   * mk * h
         + (-2.0f*s3 + 3.0f*s2)       * pk1
         + (      s3 - s2)             * mk1 * h;
}

__kernel void adjustExposureLinear(__global float4* pixels,
                                    int   w,
                                    int   h,
                                    float globalEv,
                                    float blacksEv,
                                    float shadowsEv,
                                    float highlightsEv,
                                    float whitesEv)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 px = pixels[y * w + x];

    // Zone lookup uses perceptual L: gamma-encode the linear luma so zone
    // midpoints stay where the UI/curve widget places them.
    float linLum = linear_luma(px);
    float L      = linear_to_srgb(linLum);

    float ev       = globalEv + zoneEvLinear(L, blacksEv, shadowsEv, highlightsEv, whitesEv);
    float evFactor = native_exp2(ev);

    // Exposure adjustment is a plain scale in linear light; no clamp.
    pixels[y * w + x] = (float4)(px.x * evFactor,
                                 px.y * evFactor,
                                 px.z * evFactor,
                                 1.0f);
}
)CL";

// ============================================================================
// ExposureEffect
// ============================================================================

ExposureEffect::ExposureEffect()
    : controlsWidget(nullptr), exposureParam(nullptr),
      whitesParam(nullptr), highlightsParam(nullptr),
      shadowsParam(nullptr), blacksParam(nullptr) {
}

ExposureEffect::~ExposureEffect() {
}

QString ExposureEffect::getName() const { return "Exposure"; }
QString ExposureEffect::getDescription() const {
    return "Adjusts overall image exposure with tonal zone controls";
}
QString ExposureEffect::getVersion() const { return "1.0.0"; }

bool ExposureEffect::initialize() {
    qDebug() << "Exposure effect initialized";
    return true;
}

QWidget* ExposureEffect::createControlsWidget() {
    if (controlsWidget) return controlsWidget;

    controlsWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(controlsWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    // Tone curve graph — sits at the top; updated via setParams() on every drag
    auto* curve = new ToneCurveWidget();
    curve->setToolTip("Live tone curve showing the combined effect of all zone sliders.\nThe white line is the output response; the dashed line is neutral (no adjustment).\nZone bands from left to right: Blacks, Shadows, Highlights, Whites.");
    layout->addWidget(curve);

    // Helper: re-reads all slider values and pushes them to the curve widget
    auto refreshCurve = [this, curve]() {
        ZoneEvs z;
        z.global     = float(exposureParam    ? exposureParam->value()    : 0.0);
        z.whites     = float(whitesParam      ? whitesParam->value()      : 0.0);
        z.highlights = float(highlightsParam  ? highlightsParam->value()  : 0.0);
        z.shadows    = float(shadowsParam     ? shadowsParam->value()     : 0.0);
        z.blacks     = float(blacksParam      ? blacksParam->value()      : 0.0);
        curve->setParams(z);
    };

    auto connectSlider = [&](ParamSlider* s) {
        connect(s, &ParamSlider::editingFinished, this, [this]() { emit parametersChanged(); });
        connect(s, &ParamSlider::valueChanged, curve, [=](double) {
            refreshCurve();
            emit liveParametersChanged();
        });
    };

    // Global exposure — range ±5 EV, 0.1 steps
    exposureParam = new ParamSlider("Exposure (EV)", -5.0, 5.0, 0.1, 1);
    exposureParam->setToolTip("Overall exposure offset in stops. +1 EV = one stop brighter, −1 EV = one stop darker.\nApplied in linear light (sRGB decode → scale → re-encode).");
    connectSlider(exposureParam);
    layout->addWidget(exposureParam);

    // Tonal zone offsets — range ±3 EV, 0.1 steps (bright → dark order)
    whitesParam = new ParamSlider("Whites", -3.0, 3.0, 0.1, 1);
    whitesParam->setToolTip("Exposure adjustment for the brightest tones (top 15% of the tonal range).\nPull down to recover blown highlights; push up to brighten paper whites.");
    connectSlider(whitesParam);
    layout->addWidget(whitesParam);

    highlightsParam = new ParamSlider("Highlights", -3.0, 3.0, 0.1, 1);
    highlightsParam->setToolTip("Exposure adjustment for the upper midtones and bright areas (50–85% brightness).\nUse to manage sky or facial highlight detail.");
    connectSlider(highlightsParam);
    layout->addWidget(highlightsParam);

    shadowsParam = new ParamSlider("Shadows", -3.0, 3.0, 0.1, 1);
    shadowsParam->setToolTip("Exposure adjustment for the lower midtones and darker areas (15–50% brightness).\nLift to open up shadow detail without affecting highlights.");
    connectSlider(shadowsParam);
    layout->addWidget(shadowsParam);

    blacksParam = new ParamSlider("Blacks", -3.0, 3.0, 0.1, 1);
    blacksParam->setToolTip("Exposure adjustment for the darkest tones (bottom 15% of the tonal range).\nPull down to deepen blacks and add contrast; lift to reveal shadow texture.");
    connectSlider(blacksParam);
    layout->addWidget(blacksParam);

    layout->addStretch();
    return controlsWidget;
}

QMap<QString, QVariant> ExposureEffect::getParameters() const {
    QMap<QString, QVariant> params;
    params["exposure"]   = exposureParam   ? exposureParam->value()   : 0.0;
    params["whites"]     = whitesParam     ? whitesParam->value()     : 0.0;
    params["highlights"] = highlightsParam ? highlightsParam->value() : 0.0;
    params["shadows"]    = shadowsParam    ? shadowsParam->value()    : 0.0;
    params["blacks"]     = blacksParam     ? blacksParam->value()     : 0.0;
    return params;
}

// ============================================================================
// IGpuEffect — shared pipeline interface
// ============================================================================

bool ExposureEffect::initGpuKernels(cl::Context& ctx, cl::Device& dev) {
    try {
        cl::Program prog(ctx, PIPELINE_KERNEL_SOURCE);
        prog.build({dev});
        m_kernelLinear = cl::Kernel(prog, "adjustExposureLinear");
        return true;
    }
    // GCOVR_EXCL_START
    catch (const cl::Error& e) {
        qWarning() << "[GpuPipeline] Exposure initGpuKernels failed:"
                   << e.what() << "(err" << e.err() << ")";
        return false;
    }
    // GCOVR_EXCL_STOP
}

bool ExposureEffect::enqueueGpu(cl::CommandQueue& queue,
                                 cl::Buffer& buf, cl::Buffer& /*aux*/,
                                 int w, int h,
                                 const QMap<QString, QVariant>& params) {
    const float globalEv     = float(params.value("exposure",   0.0).toDouble());
    const float blacksEv     = float(params.value("blacks",     0.0).toDouble());
    const float shadowsEv    = float(params.value("shadows",    0.0).toDouble());
    const float highlightsEv = float(params.value("highlights", 0.0).toDouble());
    const float whitesEv     = float(params.value("whites",     0.0).toDouble());

    if (globalEv == 0.0f && blacksEv == 0.0f && shadowsEv == 0.0f
        && highlightsEv == 0.0f && whitesEv == 0.0f) {
        return true;  // no-op
    }

    m_kernelLinear.setArg(0, buf);
    m_kernelLinear.setArg(1, w);
    m_kernelLinear.setArg(2, h);
    m_kernelLinear.setArg(3, globalEv);
    m_kernelLinear.setArg(4, blacksEv);
    m_kernelLinear.setArg(5, shadowsEv);
    m_kernelLinear.setArg(6, highlightsEv);
    m_kernelLinear.setArg(7, whitesEv);
    queue.enqueueNDRangeKernel(m_kernelLinear, cl::NullRange,
                               cl::NDRange(w, h), cl::NullRange);
    return true;
}

QImage ExposureEffect::processImage(const QImage& image, const QMap<QString, QVariant>& parameters) {
    if (image.isNull()) return image;

    ZoneEvs z;
    z.global     = float(parameters.value("exposure",   0.0).toDouble());
    z.whites     = float(parameters.value("whites",     0.0).toDouble());
    z.highlights = float(parameters.value("highlights", 0.0).toDouble());
    z.shadows    = float(parameters.value("shadows",    0.0).toDouble());
    z.blacks     = float(parameters.value("blacks",     0.0).toDouble());

    if (image.format() == QImage::Format_RGBX64)
        return processImageGPU16(image, z);
    return processImageGPU(image, z);
}
