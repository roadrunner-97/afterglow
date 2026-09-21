#ifndef STACKFRAMECACHE_H
#define STACKFRAMECACHE_H

#include <QByteArray>
#include <QImage>
#include <QString>

namespace StackFrameCache {

// Fingerprints the source identity and all golden-reference settings that
// affect its developed pixels. A schema token inside the implementation
// invalidates older renders when the stack pipeline changes.
QByteArray fingerprint(const QString &sourcePath, const QByteArray &settingsSignature);

QString renderPath(const QString &sourcePath);
QString fingerprintPath(const QString &sourcePath);

// Cache entries are lossless PNGs. A render is returned only when its stored
// fingerprint exactly matches `expectedFingerprint`.
QImage load(const QString &sourcePath, const QByteArray &expectedFingerprint);
bool   store(const QString &sourcePath, const QByteArray &renderFingerprint, const QImage &render);

} // namespace StackFrameCache

#endif // STACKFRAMECACHE_H
