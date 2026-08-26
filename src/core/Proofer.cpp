// GCOVR_EXCL_START — GPU dispatch and QtConcurrent lambda; covered by
// integration via GpuPipeline tests; same exclusion pattern as ImageProcessor.

#include "Proofer.h"
#include "ProofCache.h"
#include "RawLoader.h"
#include "ICropSource.h"

#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>
#include <QFile>
#include <QImageReader>
#include <QDebug>

namespace {

static QMap<QString, QVariant> buildCropInjection(ICropSource *src) {
    if (!src) return {};
    const QRectF r = src->userCropRect();
    return {
        {"_userCropX0", r.left()},
        {"_userCropY0", r.top()},
        {"_userCropX1", r.right()},
        {"_userCropY1", r.bottom()},
        {"_userCropAngle", static_cast<double>(src->userCropAngle())},
    };
}

static void mergeInto(QMap<QString, QVariant> &dst, const QMap<QString, QVariant> &src) {
    for (auto it = src.constBegin(); it != src.constEnd(); ++it) dst.insert(it.key(), it.value());
}

static QVector<GpuPipelineCall> buildGpuCalls(const EffectManager &effects) {
    const QMap<QString, QVariant> cropInjected = buildCropInjection(effects.activeCropSource());
    QVector<GpuPipelineCall>      calls;
    calls.reserve(effects.entries().size());
    for (const EffectEntry &entry : effects.entries()) {
        if (!entry.enabled || !entry.gpu) continue;
        QMap<QString, QVariant> params = entry.effect->getParameters();
        mergeInto(params, cropInjected);
        calls.append({entry.effect, entry.gpu, params});
    }
    return calls;
}

QImage scaleProof(const QImage &img) {
    constexpr int MAX_LONG_EDGE = 4096;
    const int     longEdge      = qMax(img.width(), img.height());
    if (longEdge <= MAX_LONG_EDGE) return img;
    const QSize scaled = img.size().scaled(MAX_LONG_EDGE, MAX_LONG_EDGE, Qt::KeepAspectRatio);
    return img.scaled(scaled, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
}

} // namespace

Proofer::Proofer(std::unique_ptr<EffectManager> effects, SettingsImporter::Settings defaults, ProofCache *cache,
                 QObject *parent)
    : QObject(parent), m_effects(std::move(effects)), m_defaults(std::move(defaults)), m_cache(cache),
      m_pipeline(std::make_shared<GpuPipeline>()) {}

Proofer::~Proofer() = default;

void Proofer::setQueue(QStringList paths) {
    ++m_queueGeneration;
    m_queue = std::move(paths);
    dispatchNext();
}

void Proofer::promote(const QString &path) {
    if (!m_queue.isEmpty() && m_queue.first() == path) return;
    m_queue.removeOne(path); // no-op if absent; move to front if present
    m_queue.prepend(path);
    dispatchNext();
}

void Proofer::refresh(const QString &path) {
    ++m_pathGenerations[path];
    m_queue.removeAll(path);
    m_queue.prepend(path);
    dispatchNext();
}

void Proofer::pause() {
    m_paused = true;
}

void Proofer::resume() {
    m_paused = false;
    dispatchNext();
}

void Proofer::clear() {
    ++m_queueGeneration;
    m_queue.clear();
}

void Proofer::dispatchNext() {
    if (m_paused || m_busy || m_queue.isEmpty()) return;

    const QString    path             = m_queue.takeFirst();
    const uint64_t   queueGeneration  = m_queueGeneration;
    const uint64_t   pathGeneration   = m_pathGenerations.value(path);
    const QByteArray inputFingerprint = ProofCache::inputFingerprint(path);
    m_busy                            = true;
    emit proofStarted(path);

    SettingsImporter::applyToManager(m_defaults, *m_effects);
    const QString sidecar = ProofCache::sidecarPath(path);
    if (QFile::exists(sidecar)) {
        SettingsImporter::Settings parsed;
        QString                    err;
        if (SettingsImporter::readYaml(sidecar, &parsed, &err)) SettingsImporter::applyToManager(parsed, *m_effects);
        else qWarning() << "Proofer: sidecar parse failed for" << sidecar << ":" << err;
    }

    QVector<GpuPipelineCall> calls = buildGpuCalls(*m_effects);

    auto *watcher = new QFutureWatcher<QImage>(this);
    connect(watcher, &QFutureWatcher<QImage>::finished, this,
            [this, watcher, path, queueGeneration, pathGeneration, inputFingerprint]() {
                QImage     result       = watcher->result();
                const bool currentQueue = queueGeneration == m_queueGeneration;
                const bool currentPath  = pathGeneration == m_pathGenerations.value(path);
                const bool currentInput = inputFingerprint == ProofCache::inputFingerprint(path);
                if (!result.isNull() && currentQueue && currentPath && currentInput) {
                    m_cache->store(path, result);
                    emit proofFinished(path, result);
                } else if (result.isNull() && currentQueue && currentPath) {
                    emit proofFailed(path, QString("Pipeline returned null for %1").arg(path));
                } else if (currentQueue && (!currentPath || !currentInput)) {
                    m_queue.removeAll(path);
                    m_queue.prepend(path);
                }
                m_busy = false;
                watcher->deleteLater();
                dispatchNext();
            });

    const bool isRaw    = RawLoader::isRawFile(path);
    auto       pipeline = m_pipeline;
    watcher->setFuture(QtConcurrent::run([path, calls = std::move(calls), pipeline, isRaw]() -> QImage {
        QImage img;
        if (isRaw) img = RawLoader::load(path);
        if (img.isNull()) {
            QImageReader reader(path);
            reader.setAutoTransform(true);
            img = reader.read();
        }
        if (img.isNull()) return {};
        QImage result = pipeline->run(img, calls, {}, RunMode::Commit).image;
        if (result.isNull()) return {};
        return scaleProof(result);
    }));
}

// GCOVR_EXCL_STOP
