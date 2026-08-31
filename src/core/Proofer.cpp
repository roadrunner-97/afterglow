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
#include <QPainter>
#include <QTransform>
#include <QDebug>
#include <cmath>

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

static const SettingsImporter::EffectSettings *findSettings(const SettingsImporter::Settings &settings,
                                                            const EffectEntry                &entry) {
    const QString id   = entry.effect->getId();
    const QString name = entry.effect->getName();
    for (const auto &candidate : settings.effects) {
        if ((!id.isEmpty() && candidate.id == id) || (candidate.id.isEmpty() && candidate.name == name))
            return &candidate;
    }
    return nullptr;
}

static QVector<GpuPipelineCall> buildGpuCalls(const EffectManager              &effects,
                                              const SettingsImporter::Settings &effectiveSettings) {
    const QMap<QString, QVariant> cropInjected = buildCropInjection(effects.activeCropSource());
    QVector<GpuPipelineCall>      calls;
    calls.reserve(effects.entries().size());
    for (const EffectEntry &entry : effects.entries()) {
        if (!entry.enabled || !entry.gpu) continue;
        // Proofer effects are headless: their controls widgets do not exist,
        // while many effects keep UI values in those widgets. Read the
        // authoritative saved parameters directly instead of asking the
        // headless effect object, which would return constructor defaults.
        const auto             *saved  = findSettings(effectiveSettings, entry);
        QMap<QString, QVariant> params = saved ? saved->parameters : entry.effect->getParameters();
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

QImage applyActiveCropAndRotate(const QImage &image, const QRectF &crop, float angle) {
    if (image.isNull()) return image;
    const QSize dstSize(std::max(1, static_cast<int>(std::round(crop.width() * image.width()))),
                        std::max(1, static_cast<int>(std::round(crop.height() * image.height()))));

    QImage dst(dstSize, image.format());
    dst.fill(Qt::black);
    QTransform transform;
    transform.translate(dstSize.width() * 0.5, dstSize.height() * 0.5);
    transform.rotate(-static_cast<double>(angle));
    transform.translate(-crop.center().x() * image.width(), -crop.center().y() * image.height());
    QPainter painter(&dst);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.setTransform(transform);
    painter.drawImage(0, 0, image);
    return dst;
}

} // namespace

Proofer::Proofer(std::unique_ptr<EffectManager> effects, SettingsImporter::Settings defaults, ProofCache *cache,
                 QObject *parent)
    : QObject(parent), m_effects(std::move(effects)), m_defaults(std::move(defaults)), m_cache(cache),
      m_pipeline(std::make_shared<GpuPipeline>()) {}

Proofer::~Proofer() = default;

void Proofer::setDefaults(SettingsImporter::Settings defaults) {
    m_defaults = std::move(defaults);
}

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

    SettingsImporter::Settings effectiveSettings = m_defaults;
    SettingsImporter::applyToManager(m_defaults, *m_effects);
    const QString sidecar = ProofCache::sidecarPath(path);
    if (QFile::exists(sidecar)) {
        SettingsImporter::Settings parsed;
        QString                    err;
        if (SettingsImporter::readYaml(sidecar, &parsed, &err)) {
            SettingsImporter::applyToManager(parsed, *m_effects);
            // Canonical sidecars contain all effects. For compatibility with
            // older/partial files, replace only entries they actually name.
            for (const auto &overrideSettings : parsed.effects) {
                bool replaced = false;
                for (auto &base : effectiveSettings.effects) {
                    const bool idMatch   = !overrideSettings.id.isEmpty() && overrideSettings.id == base.id;
                    const bool nameMatch = overrideSettings.id.isEmpty() && overrideSettings.name == base.name;
                    if (!idMatch && !nameMatch) continue;
                    base     = overrideSettings;
                    replaced = true;
                    break;
                }
                if (!replaced) effectiveSettings.effects.append(overrideSettings);
            }
            effectiveSettings.localAdjustments = parsed.localAdjustments;
        } else qWarning() << "Proofer: sidecar parse failed for" << sidecar << ":" << err;
    }

    QVector<GpuPipelineCall> calls       = buildGpuCalls(*m_effects, effectiveSettings);
    ICropSource             *cropSource  = m_effects->cropSource();
    const QRectF             activeCrop  = cropSource ? cropSource->userCropRect() : QRectF(0.0, 0.0, 1.0, 1.0);
    const float              activeAngle = cropSource ? cropSource->userCropAngle() : 0.0f;

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
    watcher->setFuture(QtConcurrent::run([path, calls = std::move(calls), pipeline, isRaw, cropSource, activeCrop,
                                          activeAngle, locals = effectiveSettings.localAdjustments]() -> QImage {
        QImage img;
        if (isRaw) img = RawLoader::load(path);
        if (img.isNull()) {
            QImageReader reader(path);
            reader.setAutoTransform(true);
            img = reader.read();
        }
        if (img.isNull()) return {};
        if (cropSource) img = cropSource->applyCommittedGeometry(img);
        img           = scaleProof(img);
        QImage result = pipeline->run(img, calls, {}, RunMode::Commit, locals).image;
        if (result.isNull()) return {};
        result = applyActiveCropAndRotate(result, activeCrop, activeAngle);
        return result;
    }));
}

// GCOVR_EXCL_STOP
