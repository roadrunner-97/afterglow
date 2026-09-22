#include "StackFrameCache.h"
#include "LinearImageIO.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageWriter>
#include <QSaveFile>

namespace {

QString cacheBaseName(const QString &sourcePath) {
    return QString::fromLatin1(
        QCryptographicHash::hash(QFileInfo(sourcePath).absoluteFilePath().toUtf8(), QCryptographicHash::Sha256)
            .toHex());
}

QString cacheDirectory(const QString &sourcePath) {
    return QFileInfo(sourcePath).absoluteDir().filePath(QStringLiteral(".afterglow/stack-frames"));
}

QString safeStrategyId(QString id) {
    for (QChar &c : id)
        if (!c.isLetterOrNumber() && c != QLatin1Char('-') && c != QLatin1Char('_')) c = QLatin1Char('_');
    return id;
}

QString blockDirectory(const QString &projectFolder, const QString &strategyId) {
    return QDir(projectFolder).filePath(QStringLiteral(".afterglow/stack-cache/") + safeStrategyId(strategyId));
}

} // namespace

namespace StackFrameCache {

QByteArray fingerprint(const QString &sourcePath, const QByteArray &settingsSignature) {
    const QFileInfo    source(sourcePath);
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData("afterglow-stack-render-v1");
    hash.addData(source.absoluteFilePath().toUtf8());
    hash.addData(QByteArray::number(source.size()));
    hash.addData(QByteArray::number(source.lastModified().toMSecsSinceEpoch()));
    hash.addData(settingsSignature);
    return hash.result();
}

QString renderPath(const QString &sourcePath) {
    return QDir(cacheDirectory(sourcePath)).filePath(cacheBaseName(sourcePath) + QStringLiteral(".png"));
}

QString fingerprintPath(const QString &sourcePath) {
    return QDir(cacheDirectory(sourcePath)).filePath(cacheBaseName(sourcePath) + QStringLiteral(".sha256"));
}

QImage load(const QString &sourcePath, const QByteArray &expectedFingerprint) {
    QFile storedFingerprint(fingerprintPath(sourcePath));
    if (!storedFingerprint.open(QIODevice::ReadOnly)) return {};
    if (storedFingerprint.readAll().trimmed() != expectedFingerprint.toHex()) return {};
    return QImage(renderPath(sourcePath));
}

bool store(const QString &sourcePath, const QByteArray &renderFingerprint, const QImage &render) {
    if (render.isNull()) return false;
    const QString directory = cacheDirectory(sourcePath);
    if (!QDir().mkpath(directory)) return false; // GCOVR_EXCL_LINE

    // Invalidate the old entry before replacing its pixels. If the process is
    // interrupted, a missing fingerprint is a safe cache miss next time.
    QFile::remove(fingerprintPath(sourcePath));
    QSaveFile imageFile(renderPath(sourcePath));
    if (!imageFile.open(QIODevice::WriteOnly)) return false; // GCOVR_EXCL_LINE
    QImageWriter writer(&imageFile, "PNG");
    writer.setCompression(6);
    if (!writer.write(render)) return false; // GCOVR_EXCL_LINE
    if (!imageFile.commit()) return false;   // GCOVR_EXCL_LINE

    QSaveFile fingerprintFile(fingerprintPath(sourcePath));
    if (!fingerprintFile.open(QIODevice::WriteOnly)) return false; // GCOVR_EXCL_LINE
    const QByteArray encodedFingerprint = renderFingerprint.toHex();
    if (fingerprintFile.write(encodedFingerprint) != encodedFingerprint.size()) return false; // GCOVR_EXCL_LINE
    return fingerprintFile.commit();
}

QByteArray blockFingerprint(const QVector<StackFrame> &frames, const QByteArray &settingsSignature,
                            const QByteArray &strategySignature) {
    QCryptographicHash hash(QCryptographicHash::Sha256);
    // Older blocks may have silently dropped frames after a GPU switch.
    // Rebuild them once instead of trusting a potentially partial composite.
    hash.addData("afterglow-stack-block-v2");
    hash.addData(settingsSignature);
    hash.addData(strategySignature);
    for (const StackFrame &frame : frames) {
        const QFileInfo source(frame.path);
        hash.addData(frame.decision == StackFrameDecision::Include ? "include" : "exclude");
        hash.addData(source.absoluteFilePath().toUtf8());
        hash.addData(QByteArray::number(source.size()));
        hash.addData(QByteArray::number(source.lastModified().toMSecsSinceEpoch()));
    }
    return hash.result();
}

QString blockRenderPath(const QString &projectFolder, const QString &strategyId, int blockIndex) {
    return QDir(blockDirectory(projectFolder, strategyId))
        .filePath(QStringLiteral("block-%1.exr").arg(blockIndex, 5, 10, QLatin1Char('0')));
}

static QString blockFingerprintPath(const QString &projectFolder, const QString &strategyId, int blockIndex) {
    return QDir(blockDirectory(projectFolder, strategyId))
        .filePath(QStringLiteral("block-%1.sha256").arg(blockIndex, 5, 10, QLatin1Char('0')));
}

QImage loadBlock(const QString &projectFolder, const QString &strategyId, int blockIndex,
                 const QByteArray &expectedFingerprint) {
    QFile storedFingerprint(blockFingerprintPath(projectFolder, strategyId, blockIndex));
    if (!storedFingerprint.open(QIODevice::ReadOnly)) return {};
    if (storedFingerprint.readAll().trimmed() != expectedFingerprint.toHex()) return {};
    return LinearImageIO::readExr(blockRenderPath(projectFolder, strategyId, blockIndex));
}

bool storeBlock(const QString &projectFolder, const QString &strategyId, int blockIndex,
                const QByteArray &renderFingerprint, const QImage &render) {
    const QString directory = blockDirectory(projectFolder, strategyId);
    if (!QDir().mkpath(directory)) return false;
    const QString fingerprint = blockFingerprintPath(projectFolder, strategyId, blockIndex);
    QFile::remove(fingerprint);
    if (!LinearImageIO::writeExr(blockRenderPath(projectFolder, strategyId, blockIndex), render)) return false;
    QSaveFile fingerprintFile(fingerprint);
    if (!fingerprintFile.open(QIODevice::WriteOnly)) return false;
    const QByteArray encoded = renderFingerprint.toHex();
    if (fingerprintFile.write(encoded) != encoded.size()) return false;
    return fingerprintFile.commit();
}

} // namespace StackFrameCache
