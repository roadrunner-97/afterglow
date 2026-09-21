#include "StackFrameCache.h"

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

} // namespace StackFrameCache
