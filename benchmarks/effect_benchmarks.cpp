#include <benchmark/benchmark.h>

#include "BlurEffect.h"
#include "BrightnessEffect.h"
#include "ClarityEffect.h"
#include "ColorBalanceEffect.h"
#include "CropRotateEffect.h"
#include "DenoiseEffect.h"
#include "EffectManager.h"
#include "ExposureEffect.h"
#include "FilmGrainEffect.h"
#include "GpuDeviceRegistry.h"
#include "GpuDeviceRegistryOCL.h"
#include "GpuPipeline.h"
#include "GrayscaleEffect.h"
#include "HotPixelEffect.h"
#include "IGpuEffect.h"
#include "ImageProcessor.h"
#include "SaturationEffect.h"
#include "SplitToningEffect.h"
#include "UnsharpEffect.h"
#include "VignetteEffect.h"
#include "WhiteBalanceEffect.h"

#include <QApplication>
#include <QCheckBox>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QThreadPool>
#include <QTimer>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kBurstSize = 12;

using EffectFactory = std::function<std::unique_ptr<PhotoEditorEffect>()>;

struct EffectSpec {
    const char             *name;
    EffectFactory           factory;
    QMap<QString, QVariant> parameters;
    const char             *varyKey;
    double                  varyStep;
};

template <typename T> EffectFactory factory() {
    return [] { return std::make_unique<T>(); };
}

std::vector<EffectSpec> effectSpecs() {
    return {
        {"CropRotate",
         factory<CropRotateEffect>(),
         {{"cropX0", 0.05}, {"cropY0", 0.05}, {"cropX1", 0.95}, {"cropY1", 0.95}, {"angle", 2.0}},
         "angle",
         0.1},
        {"HotPixel", factory<HotPixelEffect>(), {{"threshold", 30.0}}, "threshold", 1.0},
        {"Exposure",
         factory<ExposureEffect>(),
         {{"exposure", 0.7}, {"blacks", 0.2}, {"shadows", 0.3}, {"highlights", -0.3}, {"whites", -0.2}},
         "exposure",
         0.05},
        {"WhiteBalance",
         factory<WhiteBalanceEffect>(),
         {{"shot_temp", 5500.0}, {"temperature", 6800.0}, {"tint", 12.0}},
         "temperature",
         25.0},
        {"Brightness", factory<BrightnessEffect>(), {{"brightness", 18.0}, {"contrast", 12.0}}, "brightness", 1.0},
        {"Saturation", factory<SaturationEffect>(), {{"saturation", 22.0}, {"vibrancy", 18.0}}, "saturation", 1.0},
        {"Blur", factory<BlurEffect>(), {{"blurType", 0}, {"radius", 12}}, "radius", 1.0},
        {"Grayscale", factory<GrayscaleEffect>(), {{"active", true}}, "active", 0.0},
        {"Unsharp", factory<UnsharpEffect>(), {{"amount", 1.2}, {"radius", 4}, {"threshold", 3}}, "amount", 0.05},
        {"Denoise",
         factory<DenoiseEffect>(),
         {{"strength", 45.0}, {"shadowPreserve", 50.0}, {"colorNoise", 40.0}, {"algorithm", 0}},
         "strength",
         1.0},
        {"Vignette",
         factory<VignetteEffect>(),
         {{"amount", -35.0},
          {"midpoint", 45.0},
          {"feather", 60.0},
          {"roundness", 0.0},
          {"centerX", 50.0},
          {"centerY", 50.0}},
         "amount",
         1.0},
        {"FilmGrain",
         factory<FilmGrainEffect>(),
         {{"amount", 24.0}, {"size", 1.5}, {"lumWeight", true}, {"seed", 7}},
         "amount",
         1.0},
        {"SplitToning",
         factory<SplitToningEffect>(),
         {{"shadowHue", 220.0}, {"shadowSat", 25.0}, {"highlightHue", 42.0}, {"highlightSat", 20.0}, {"balance", 5.0}},
         "balance",
         1.0},
        {"Clarity", factory<ClarityEffect>(), {{"amount", 35.0}, {"radius", 18}}, "amount", 1.0},
        {"ColorBalance",
         factory<ColorBalanceEffect>(),
         {{"shadowR", -8.0},
          {"shadowG", 2.0},
          {"shadowB", 10.0},
          {"midtoneR", 5.0},
          {"midtoneG", 0.0},
          {"midtoneB", -4.0},
          {"highlightR", 8.0},
          {"highlightG", 2.0},
          {"highlightB", -6.0}},
         "midtoneR",
         1.0},
    };
}

QImage makeImage(int w, int h) {
    QImage image(w, h, QImage::Format_RGB32);
    for (int y = 0; y < h; ++y) {
        auto *row = reinterpret_cast<QRgb *>(image.scanLine(y));
        for (int x = 0; x < w; ++x) {
            row[x] = qRgb((x * 255) / std::max(1, w - 1), (y * 255) / std::max(1, h - 1),
                          ((x + y) * 255) / std::max(1, w + h - 2));
        }
    }
    return image;
}

std::unique_ptr<PhotoEditorEffect> makeConfiguredEffect(const EffectSpec &spec) {
    auto effect = spec.factory();
    if (!effect->initialize()) throw std::runtime_error(std::string("Failed to initialize ") + spec.name);
    effect->applyParameters(spec.parameters);
    // Grayscale predates parameter serialization and stores its checkbox state
    // internally, so exercise the real UI binding once to enable its kernel.
    if (QString::fromLatin1(spec.name) == QStringLiteral("Grayscale")) {
        std::unique_ptr<QWidget> controls(effect->createControlsWidget());
        if (auto *check = controls->findChild<QCheckBox *>()) check->setChecked(true);
    }
    return effect;
}

QMap<QString, QVariant> pipelineParameters(const EffectSpec &spec, int sourceW, int sourceH, double variation = 0.0) {
    QMap<QString, QVariant> params = spec.parameters;
    if (spec.varyKey && spec.varyStep != 0.0)
        params.insert(QString::fromLatin1(spec.varyKey),
                      params.value(QString::fromLatin1(spec.varyKey)).toDouble() + variation * spec.varyStep);
    params.insert("_srcPixelsPerPreviewPixel", 1.0);
    params.insert("_cropX0", 0.0);
    params.insert("_cropY0", 0.0);
    params.insert("_srcW", sourceW);
    params.insert("_srcH", sourceH);
    return params;
}

QVector<GpuPipelineCall> singleCall(PhotoEditorEffect *effect, const EffectSpec &spec, int sourceW, int sourceH,
                                    double variation = 0.0) {
    auto *gpu = dynamic_cast<IGpuEffect *>(effect);
    if (!gpu) throw std::runtime_error(std::string(spec.name) + " has no GPU implementation");
    return {{effect, gpu, pipelineParameters(spec, sourceW, sourceH, variation)}};
}

void setRateCounters(benchmark::State &state, int w, int h) {
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(w) * static_cast<int64_t>(h));
    state.counters["MPixels/s"] =
        benchmark::Counter(static_cast<double>(w) * h / 1.0e6, benchmark::Counter::kIsIterationInvariantRate);
}

void benchmarkKernelOnly(benchmark::State &state, EffectSpec spec) {
    const int    width  = static_cast<int>(state.range(0));
    const int    height = static_cast<int>(state.range(1));
    cl::Device   device;
    cl::Platform platform;
    if (!GpuDeviceRegistryOCL::getSelectedDevice(device, platform)) {
        state.SkipWithError("No OpenCL device");
        return;
    }
    cl::Context      context(device);
    cl::CommandQueue queue(context, device);
    auto             effect = makeConfiguredEffect(spec);
    auto            *gpu    = dynamic_cast<IGpuEffect *>(effect.get());
    if (!gpu || !gpu->initGpuKernels(context, device)) {
        state.SkipWithError("Effect kernel initialization failed");
        return;
    }

    const size_t    bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(cl_float4);
    cl::Buffer      buf(context, CL_MEM_READ_WRITE, bytes);
    cl::Buffer      aux(context, CL_MEM_READ_WRITE, bytes);
    const cl_float4 fill = {{0.18f, 0.35f, 0.62f, 1.0f}};
    queue.enqueueFillBuffer(buf, fill, 0, bytes);
    queue.finish();
    const auto params = pipelineParameters(spec, width, height);

    for (auto _ : state) {
        (void)_;
        if (!gpu->enqueueGpu(queue, buf, aux, width, height, params)) {
            state.SkipWithError("enqueueGpu failed");
            break;
        }
        queue.finish();
        benchmark::ClobberMemory();
    }
    setRateCounters(state, width, height);
}

void benchmarkTransfer(benchmark::State &state, EffectSpec spec) {
    const int             sourceW  = static_cast<int>(state.range(0));
    const int             sourceH  = static_cast<int>(state.range(1));
    const int             previewW = std::max(1, sourceW / 2);
    const int             previewH = std::max(1, sourceH / 2);
    auto                  effect   = makeConfiguredEffect(spec);
    GpuPipeline           pipeline;
    QImage                image = makeImage(sourceW, sourceH);
    const ViewportRequest viewport{{previewW, previewH}, 1.0f, {0.5, 0.5}};
    auto                  calls = singleCall(effect.get(), spec, sourceW, sourceH);
    if (pipeline.run(image, calls, viewport, RunMode::LiveDrag).image.isNull()) {
        state.SkipWithError("Pipeline warm-up failed");
        return;
    }

    int mutation = 0;
    for (auto _ : state) {
        (void)_;
        auto *row         = reinterpret_cast<QRgb *>(image.scanLine(0));
        row[0]            = qRgb(mutation++ & 255, 127, 63); // detach/cacheKey change forces a new upload
        const auto result = pipeline.run(image, calls, viewport, RunMode::LiveDrag);
        benchmark::DoNotOptimize(result.image.constBits());
        if (result.image.isNull()) {
            state.SkipWithError("Pipeline run failed");
            break;
        }
    }
    setRateCounters(state, previewW, previewH);
    state.counters["UploadMB"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * sourceW * sourceH * 4.0 / 1.0e6, benchmark::Counter::kAvgIterations);
    state.counters["DownloadMB"] =
        benchmark::Counter(static_cast<double>(state.iterations()) * previewW * previewH * 4.0 / 1.0e6,
                           benchmark::Counter::kAvgIterations);
}

bool runAndWait(ImageProcessor &processor, int timeoutMs, int &delivered, const std::function<void()> &submit) {
    QEventLoop loop;
    bool       completed  = false;
    const auto connection = QObject::connect(&processor, &ImageProcessor::processingComplete, &loop,
                                             [&](const QImage &image, const QPoint &) {
                                                 ++delivered;
                                                 completed = !image.isNull();
                                                 loop.quit();
                                             });
    QTimer::singleShot(timeoutMs, &loop, &QEventLoop::quit);
    submit();
    loop.exec();
    QObject::disconnect(connection);
    QThreadPool::globalInstance()->waitForDone();
    QApplication::processEvents();
    return completed;
}

void benchmarkRepeated(benchmark::State &state, EffectSpec spec) {
    const int     sourceW  = static_cast<int>(state.range(0));
    const int     sourceH  = static_cast<int>(state.range(1));
    const int     previewW = std::max(1, sourceW / 2);
    const int     previewH = std::max(1, sourceH / 2);
    EffectManager manager;
    manager.addEffect(makeConfiguredEffect(spec));
    ImageProcessor        processor;
    const QImage          image = makeImage(sourceW, sourceH);
    const ViewportRequest viewport{{previewW, previewH}, 1.0f, {0.5, 0.5}};

    int warmDelivered = 0;
    if (!runAndWait(processor, 30000, warmDelivered,
                    [&] { processor.processImageAsync(image, manager, viewport, RunMode::LiveDrag); })) {
        state.SkipWithError("Asynchronous pipeline warm-up failed");
        return;
    }

    int64_t totalDelivered = 0;
    for (auto _ : state) {
        (void)_;
        int        delivered = 0;
        const bool completed = runAndWait(processor, 30000, delivered, [&] {
            for (int request = 0; request < kBurstSize; ++request) {
                QMap<QString, QVariant> params = spec.parameters;
                if (spec.varyKey && spec.varyStep != 0.0) {
                    const QString key = QString::fromLatin1(spec.varyKey);
                    params.insert(key, params.value(key).toDouble() + request * spec.varyStep);
                    manager.entries().front().effect->applyParameters(params);
                }
                processor.processImageAsync(image, manager, viewport, RunMode::LiveDrag);
            }
        });
        if (!completed) {
            state.SkipWithError("Timed out waiting for newest generation");
            break;
        }
        totalDelivered += delivered;
    }
    state.counters["Submitted"] =
        benchmark::Counter(static_cast<double>(state.iterations() * kBurstSize), benchmark::Counter::kAvgIterations);
    state.counters["Delivered"] =
        benchmark::Counter(static_cast<double>(totalDelivered), benchmark::Counter::kAvgIterations);
}

struct EffectStack {
    std::vector<std::unique_ptr<PhotoEditorEffect>> effects;
    QVector<GpuPipelineCall>                        calls;
};

EffectStack makeStack(int sourceW, int sourceH) {
    EffectStack stack;
    for (const auto &spec : effectSpecs()) {
        auto  effect = makeConfiguredEffect(spec);
        auto *gpu    = dynamic_cast<IGpuEffect *>(effect.get());
        auto  params = pipelineParameters(spec, sourceW, sourceH);
        params.insert("_userCropX0", 0.05);
        params.insert("_userCropY0", 0.05);
        params.insert("_userCropX1", 0.95);
        params.insert("_userCropY1", 0.95);
        params.insert("_userCropAngle", 2.0);
        stack.calls.append({effect.get(), gpu, std::move(params)});
        stack.effects.push_back(std::move(effect));
    }
    return stack;
}

void benchmarkAllEffects(benchmark::State &state, RunMode mode) {
    const int             sourceW  = static_cast<int>(state.range(0));
    const int             sourceH  = static_cast<int>(state.range(1));
    const int             previewW = std::max(1, sourceW / 2);
    const int             previewH = std::max(1, sourceH / 2);
    auto                  stack    = makeStack(sourceW, sourceH);
    GpuPipeline           pipeline;
    const QImage          image = makeImage(sourceW, sourceH);
    const ViewportRequest viewport{{previewW, previewH}, 1.0f, {0.5, 0.5}};

    if (mode == RunMode::PanZoom) {
        if (pipeline.run(image, stack.calls, viewport, RunMode::Commit).image.isNull()) {
            state.SkipWithError("Commit warm-up failed");
            return;
        }
    } else if (pipeline.run(image, stack.calls, viewport, mode).image.isNull()) {
        state.SkipWithError("Pipeline warm-up failed");
        return;
    }

    for (auto _ : state) {
        (void)_;
        const auto result = pipeline.run(image, stack.calls, viewport, mode);
        benchmark::DoNotOptimize(result.image.constBits());
        if (result.image.isNull()) {
            state.SkipWithError("Pipeline run failed");
            break;
        }
    }
    setRateCounters(state, mode == RunMode::Commit ? sourceW : previewW, mode == RunMode::Commit ? sourceH : previewH);
}

benchmark::Benchmark *atResolutions(benchmark::Benchmark *bench) {
    return bench->Args({1920, 1080})->Args({3840, 2160})->Args({6000, 4000});
}

void registerBenchmarks() {
    for (const auto &spec : effectSpecs()) {
        atResolutions(
            benchmark::RegisterBenchmark((std::string("KernelOnly/") + spec.name).c_str(), benchmarkKernelOnly, spec)
                ->Unit(benchmark::kMillisecond)
                ->UseRealTime());
        atResolutions(
            benchmark::RegisterBenchmark((std::string("Transfer/") + spec.name).c_str(), benchmarkTransfer, spec)
                ->Unit(benchmark::kMillisecond)
                ->UseRealTime());
        atResolutions(
            benchmark::RegisterBenchmark((std::string("Repeated/") + spec.name).c_str(), benchmarkRepeated, spec)
                ->Unit(benchmark::kMillisecond)
                ->UseRealTime()
                ->Iterations(3));
    }
    atResolutions(benchmark::RegisterBenchmark("AllEffects/LivePreview", benchmarkAllEffects, RunMode::LiveDrag)
                      ->Unit(benchmark::kMillisecond)
                      ->UseRealTime());
    atResolutions(benchmark::RegisterBenchmark("AllEffects/Commit", benchmarkAllEffects, RunMode::Commit)
                      ->Unit(benchmark::kMillisecond)
                      ->UseRealTime());
    atResolutions(benchmark::RegisterBenchmark("AllEffects/CachedPanZoom", benchmarkAllEffects, RunMode::PanZoom)
                      ->Unit(benchmark::kMillisecond)
                      ->UseRealTime());
}

} // namespace

int main(int argc, char **argv) {
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    GpuDeviceRegistry::instance().enumerate();
    if (GpuDeviceRegistry::instance().count() == 0) {
        qCritical("No OpenCL devices found");
        return 1;
    }
    const auto &device = GpuDeviceRegistry::instance().devices().front();
    qInfo().noquote() << "Benchmark OpenCL device:" << device.name << "(" << device.typeName << ") on"
                      << device.platformName;

    registerBenchmarks();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
