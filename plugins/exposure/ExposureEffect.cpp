#include "ExposureEffect.h"
#include "ParamSlider.h"
#include "color_kernels.h"
#include <QDebug>
#include <QVBoxLayout>
#include <QPainter>
#include <QPolygonF>
#include <QSizePolicy>
#include <QMouseEvent>
#include <functional>
#include <cmath>

namespace {

struct ZoneEvs {
    float global, blacks, shadows, highlights, whites;
};


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
    // Zone adjustments only — global EV is reflected by shifting the
    // histogram overlay, not by compressing this curve.  Keeping the curve
    // anchored at fixed zone positions preserves diagnostic detail: the
    // inflection points always sit at the labelled zone bands, and the
    // shifted histogram shows where pixels actually land post-exposure.
    float zEv = cpuZoneEv(L, z);
    return cpuLinearToSrgb(cpuSrgbToLinear(L) * std::exp2(zEv));
}

// Zone midpoints on the input luminance axis
static constexpr float ZONE_X[4] = { 0.075f, 0.325f, 0.675f, 0.925f };

class ToneCurveWidget : public QWidget {
public:
    explicit ToneCurveWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        setFixedHeight(180);
        setMouseTracking(true);
    }

    void setParams(const ZoneEvs& z) { m_z = z; update(); }
    void setHistogram(const std::vector<uint32_t>& h) { m_histogram = h; update(); }

    // callbacks wired by ExposureEffect::createControlsWidget.
    // zone: 0=blacks 1=shadows 2=highlights 3=whites.
    std::function<void(int zone, float ev)> onZoneChanged;
    std::function<void(float ev)>           onGlobalChanged;
    std::function<void()>                   onEditingFinished;

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        const QRectF r = plotRect();
        const float W = r.width(), H = r.height();
        auto toScreen = [&](float L, float out) -> QPointF {
            return { r.left() + L * W, r.bottom() - out * H };
        };

        // Background gradient
        QLinearGradient bg(r.left(), 0, r.right(), 0);
        bg.setColorAt(0.0, QColor(20, 20, 20));
        bg.setColorAt(1.0, QColor(55, 55, 55));
        p.fillRect(r, bg);

        // Zone bands
        struct ZoneBand { float x0, x1; QColor col; };
        const ZoneBand bands[] = {
            { 0.00f, 0.15f, QColor( 91, 110, 168, 32) },
            { 0.15f, 0.50f, QColor( 58, 136, 152, 26) },
            { 0.50f, 0.85f, QColor(192, 128,  44, 26) },
            { 0.85f, 1.00f, QColor(200, 168,  64, 32) },
        };
        for (const auto& b : bands) {
            QRectF br(r.left() + b.x0 * W, r.top(), (b.x1 - b.x0) * W, H);
            p.fillRect(br, b.col);
        }

        // Grid lines
        p.setPen(QPen(QColor(80, 80, 80, 140), 1));
        for (float frac : { 0.15f, 0.50f, 0.85f })
            p.drawLine(toScreen(frac, 0.0f), toScreen(frac, 1.0f));
        for (int i = 1; i <= 3; i++)
            p.drawLine(toScreen(0.0f, float(i) / 4.0f), toScreen(1.0f, float(i) / 4.0f));

        // Zone labels
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

        // histogram shifted by global EV. For each output x, invert
        // the EV transform to find which input bin feeds that position.
        if (!m_histogram.empty()) {
            const int nBins = int(m_histogram.size());
            const float globalFac    = std::exp2(m_z.global);
            const float invGlobalFac = 1.0f / globalFac;

            uint32_t peak = 0;
            for (uint32_t b : m_histogram) if (b > peak) peak = b;
            if (peak > 0) {

                const float invLogPk = 1.0f / std::log1p(float(peak));
                const int drawBins = std::max(2, int(W));
                QPolygonF hist;
                hist.reserve(drawBins + 2);
                hist << toScreen(0.0f, 0.0f);

                for (int i = 0; i < drawBins; ++i) {

                    float xL  = (i + 0.5f) / float(drawBins);
                    // Input luminance that maps to xL after global EV
                    float inL = cpuLinearToSrgb(cpuSrgbToLinear(xL) * invGlobalFac);
                    inL = std::max(0.0f, std::min(1.0f, inL));
                    int srcBin = std::max(0, std::min(nBins - 1, int(inL * float(nBins))));
                    float hN  = std::log1p(float(m_histogram[srcBin])) * invLogPk;
                    hist << toScreen(xL, hN);
                }
                hist << toScreen(1.0f, 0.0f);
                p.setPen(Qt::NoPen);
                p.setBrush(QColor(230, 230, 230, 90));
                p.drawPolygon(hist);
            }
        }

        // Identity diagonal
        p.setPen(QPen(QColor(140, 140, 140, 100), 1, Qt::DashLine));
        p.drawLine(toScreen(0.0f, 0.0f), toScreen(1.0f, 1.0f));

        // Tone curve
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

        // Zone control points
        static const QColor ptColors[4] = {
            QColor(120, 150, 220),   // blacks
            QColor( 80, 190, 200),   // shadows
            QColor(220, 160,  60),   // highlights
            QColor(220, 215, 100),   // whites
        };

        for (int i = 0; i < 4; ++i) {

            QPointF pt = zonePoint(i, r);
            bool active = (m_dragTarget == i || m_hoverTarget == i);
            p.setPen(QPen(Qt::white, active ? 2.0f : 1.5f));
            p.setBrush(active ? ptColors[i].lighter(130) : ptColors[i]);
            p.drawEllipse(pt, active ? 6.0 : 5.0, active ? 6.0 : 5.0);
        }

        // Global EV handle, the diamond on the left rail
        {
            QPointF gp = globalPoint(r);
            bool active = (m_dragTarget == 4 || m_hoverTarget == 4);
            const float hs = active ? 6.5f : 5.5f;
            QPolygonF diamond;
            diamond << QPointF(gp.x(), gp.y() - hs)
                    << QPointF(gp.x() + hs, gp.y())
                    << QPointF(gp.x(), gp.y() + hs)
                    << QPointF(gp.x() - hs, gp.y());
            p.setPen(QPen(Qt::white, active ? 2.0f : 1.5f));
            p.setBrush(active ? QColor(220, 220, 80) : QColor(180, 180, 60));
            p.drawPolygon(diamond);
            // "EV" label
            p.setPen(QColor(200, 200, 200, 160));
            QFont lf = p.font();
            lf.setPixelSize(8);
            p.setFont(lf);
            p.drawText(QRectF(r.left() + 14, gp.y() - 6, 20, 12),
                       Qt::AlignLeft | Qt::AlignVCenter, "EV");
        }

        // Border
        p.setBrush(Qt::NoBrush);
        p.setPen(QPen(QColor(90, 90, 90), 1));
        p.drawRect(r);

        // Base exposure label
        if (std::abs(m_z.global) >= 0.05f) {
            
            QString label = QString("Base: %1%2 EV")
                .arg(m_z.global > 0 ? "+" : "")
                .arg(double(m_z.global), 0, 'f', 1);
            p.setPen(QColor(200, 200, 200, 180));
            QFont lf = p.font();
            lf.setPixelSize(9);
            p.setFont(lf);
            QRectF lr(r.left() + 36, r.top() + 4, 80, 12);
            p.drawText(lr, Qt::AlignLeft | Qt::AlignVCenter, label);
        }
    }

    void mousePressEvent(QMouseEvent* e) override {

        if (e->button() != Qt::LeftButton) return;
        m_dragTarget = hitTest(e->pos());
        update();
    }

    void mouseMoveEvent(QMouseEvent* e) override {

        if (m_dragTarget < 0) {
            int h = hitTest(e->pos());
            if (h != m_hoverTarget) { m_hoverTarget = h; update(); }
            setCursor(h >= 0 ? Qt::SizeVerCursor : Qt::ArrowCursor);
            return;
        }

        QRectF r = plotRect();

        if (m_dragTarget == 4) {
            float ev = screenToGlobalEv(e->pos(), r);
            if (onGlobalChanged) onGlobalChanged(ev);
        } else {
            float ev = screenToZoneEv(m_dragTarget, e->pos(), r);
            if (onZoneChanged) onZoneChanged(m_dragTarget, ev);
        }
    }

    void mouseReleaseEvent(QMouseEvent* e) override {
        if (e->button() != Qt::LeftButton || m_dragTarget < 0) return;
        m_dragTarget = -1;
        update();
        if (onEditingFinished) onEditingFinished();
    }

    void leaveEvent(QEvent*) override {
        if (m_hoverTarget >= 0) { m_hoverTarget = -1; update(); }
        setCursor(Qt::ArrowCursor);
    }

private:
    ZoneEvs               m_z{};
    std::vector<uint32_t> m_histogram;
    int m_dragTarget  = -1;   // -1=none 0-3=zone 4=global
    int m_hoverTarget = -1;

    QRectF plotRect() const {
        return QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5);
    }

    float zoneEvValue(int i) const {
        switch (i) {
            case 0: return m_z.blacks;
            case 1: return m_z.shadows;
            case 2: return m_z.highlights;
            case 3: return m_z.whites;
            default: return 0.0f;
        }
    }

    QPointF zonePoint(int i, const QRectF& r) const {

        float L   = ZONE_X[i];
        float out = cpuCurve(L, m_z);
        return { r.left() + L * r.width(), r.bottom() - out * r.height() };
    }

    // Global EV handle: vertical rail near the left edge
    QPointF globalPoint(const QRectF& r) const {

        float frac = std::max(0.0f, std::min(1.0f, (m_z.global + 5.0f) / 10.0f));
        return { r.left() + 9.0f, r.bottom() - frac * r.height() };
    }

    int hitTest(QPointF pos) const {
        QRectF r = plotRect();

        for (int i = 0; i < 4; ++i) {
            QPointF pt = zonePoint(i, r);
            float dx = float(pos.x() - pt.x()), dy = float(pos.y() - pt.y());
            if (dx*dx + dy*dy <= 121.0f) return i;  // 11px radius
        }

        QPointF gp = globalPoint(r);
        float dx = float(pos.x() - gp.x()), dy = float(pos.y() - gp.y());
        if (dx*dx + dy*dy <= 121.0f) return 4;
        return -1;
    }

    float screenToZoneEv(int zone, QPointF pos, const QRectF& r) const {

        float out = 1.0f - float(pos.y() - r.top()) / r.height();
        out = std::max(0.001f, std::min(0.999f, out));
        float linOut = cpuSrgbToLinear(out);
        float linIn  = cpuSrgbToLinear(ZONE_X[zone]);
        if (linIn <= 0.0f) return 0.0f;
        return std::max(-3.0f, std::min(3.0f, std::log2(linOut / linIn)));
    }

    float screenToGlobalEv(QPointF pos, const QRectF& r) const {

        float frac = 1.0f - float(pos.y() - r.top()) / r.height();
        frac = std::max(0.0f, std::min(1.0f, frac));
        return std::max(-5.0f, std::min(5.0f, frac * 10.0f - 5.0f));
    }
};

} // namespace

// ============================================================================
// Pipeline kernel — float4 linear sRGB, used by GpuPipeline via enqueueGpu.
// Exposure is applied in linear light; zone selection uses perceptual L
// derived from the *post-global-EV* linear luminance, so the zone sliders act
// on the image the user sees (e.g. after raising a 3-stop-underexposed shot
// by +3 EV, newly-bright pixels now land in highlights/whites).  Outputs are
// not clamped — scene-linear values above 1.0 are valid HDR; the final pack
// kernel clamps once.
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

    // Zone lookup uses perceptual L derived from post-global-EV luminance.
    // linear_to_srgb clamps to [0,1], so linLum*globalFac above 1 lands in the
    // whites zone — which is exactly the desired behaviour for boosted
    // underexposed shots.
    float linLum    = linear_luma(px);
    float globalFac = native_exp2(globalEv);
    float L         = linear_to_srgb(linLum * globalFac);

    float zEv      = zoneEvLinear(L, blacksEv, shadowsEv, highlightsEv, whitesEv);
    float totalFac = globalFac * native_exp2(zEv);

    // Exposure adjustment is a plain scale in linear light; no clamp.
    pixels[y * w + x] = (float4)(px.x * totalFac,
                                 px.y * totalFac,
                                 px.z * totalFac,
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
    curve->setToolTip("Live tone curve showing the zone sliders' effect.\nThe white line is the zone response; the dashed line is neutral.\nThe grey fill is the luminance histogram, shifted rightward by the Exposure slider to show where pixels land on the adjusted image.\nZone bands from left to right: Blacks, Shadows, Highlights, Whites.");
    layout->addWidget(curve);

    // Bridge histogram data into the anon-namespace widget.
    m_applyHistogram = [curve](const std::vector<uint32_t>& h) { curve->setHistogram(h); };
    if (!m_histogram.empty()) curve->setHistogram(m_histogram);

    // Helper: re-reads all slider values and pushes them to the curve widget.
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

    // Graph to sliders: update the relevant slider 
    // then refresh the curve and use liveParametersChanged once.
    curve->onZoneChanged = [this, refreshCurve](int zone, float ev) {
        ParamSlider* zoneSliders[4] = { blacksParam, shadowsParam, highlightsParam, whitesParam };
        if (zoneSliders[zone]) {
            zoneSliders[zone]->blockSignals(true);
            zoneSliders[zone]->setValue(ev);
            zoneSliders[zone]->blockSignals(false);
        }
        refreshCurve();
        emit liveParametersChanged();
    };
    curve->onGlobalChanged = [this, refreshCurve](float ev) {
        if (exposureParam) {
            exposureParam->blockSignals(true);
            exposureParam->setValue(ev);
            exposureParam->blockSignals(false);
        }
        refreshCurve();
        emit liveParametersChanged();
    };
    curve->onEditingFinished = [this]() { emit parametersChanged(); };

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

void ExposureEffect::applyParameters(const QMap<QString, QVariant>& parameters) {
    auto apply = [&](ParamSlider* p, const char* key) {
        if (p && parameters.contains(key))
            p->setValue(parameters.value(key).toDouble());
    };
    apply(exposureParam,   "exposure");
    apply(whitesParam,     "whites");
    apply(highlightsParam, "highlights");
    apply(shadowsParam,    "shadows");
    apply(blacksParam,     "blacks");
    emit parametersChanged();
}

void ExposureEffect::onImageLoaded(const ImageMetadata& meta) {
    m_histogram = meta.luminanceHistogram;
    if (m_applyHistogram) m_applyHistogram(m_histogram);
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

