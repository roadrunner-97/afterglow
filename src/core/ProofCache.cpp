#include "ProofCache.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QCryptographicHash>

ProofCache::ProofCache(QObject *parent) : QObject(parent) {}

// static
QString ProofCache::proofPath(const QString &imagePath) {
    const QFileInfo fi(imagePath);
    // <folder>/.afterglow/proofs/<original-filename>.jpg
    // The full filename (including extension) disambiguates IMG_0001.CR2
    // from IMG_0001.NEF when both exist in the same folder.
    return fi.absoluteDir().filePath(".afterglow/proofs/" + fi.fileName() + ".jpg");
}

// static
QString ProofCache::sidecarPath(const QString &imagePath) {
    const QFileInfo fi(imagePath);
    return fi.absoluteDir().filePath(fi.completeBaseName() + ".yml");
}

QByteArray ProofCache::inputFingerprint(const QString &imagePath) {
    QCryptographicHash hash(QCryptographicHash::Sha256);
    const QFileInfo    sourceFi(imagePath);
    hash.addData(imagePath.toUtf8());
    hash.addData(QByteArray::number(sourceFi.size()));
    hash.addData(QByteArray::number(sourceFi.lastModified().toMSecsSinceEpoch()));

    QFile sidecar(sidecarPath(imagePath));
    if (sidecar.open(QIODevice::ReadOnly)) hash.addData(sidecar.readAll());
    return hash.result();
}

bool ProofCache::isProofed(const QString &imagePath) const {
    const QFileInfo proofFi(proofPath(imagePath));
    if (!proofFi.exists()) return false;

    const QFileInfo sourceFi(imagePath);
    if (sourceFi.exists() && proofFi.lastModified() < sourceFi.lastModified()) return false;

    const QFileInfo sidecarFi(sidecarPath(imagePath));
    if (!sidecarFi.exists()) return true; // no edits → always fresh

    return proofFi.lastModified() >= sidecarFi.lastModified();
}

QImage ProofCache::proof(const QString &imagePath) {
    if (!isProofed(imagePath)) {
        if (m_lruCache.remove(imagePath) > 0) m_lruOrder.removeOne(imagePath);
        return {};
    }

    auto it = m_lruCache.find(imagePath);
    if (it != m_lruCache.end()) {
        lruPromote(imagePath);
        return it.value();
    }

    QImage img(proofPath(imagePath));
    if (img.isNull()) return {};

    lruInsert(imagePath, img);
    return img;
}

void ProofCache::store(const QString &imagePath, const QImage &proof) {
    const QString path = proofPath(imagePath);
    QDir().mkpath(QFileInfo(path).absolutePath());
    proof.save(path, "JPEG", 90);
    lruInsert(imagePath, proof);
}

void ProofCache::invalidate(const QString &imagePath) {
    const QString path = proofPath(imagePath);
    if (QFileInfo::exists(path)) QFile::remove(path);

    if (m_lruCache.remove(imagePath) > 0) m_lruOrder.removeOne(imagePath);
}

void ProofCache::clear() {
    m_lruOrder.clear();
    m_lruCache.clear();
}

void ProofCache::lruInsert(const QString &key, const QImage &img) {
    if (m_lruCache.contains(key)) {
        lruPromote(key);
        m_lruCache[key] = img;
        return;
    }
    lruEvict();
    m_lruOrder.prepend(key);
    m_lruCache[key] = img;
}

void ProofCache::lruPromote(const QString &key) {
    m_lruOrder.removeOne(key);
    m_lruOrder.prepend(key);
}

void ProofCache::lruEvict() {
    while (m_lruOrder.size() >= MAX_LRU) {
        const QString evicted = m_lruOrder.takeLast();
        m_lruCache.remove(evicted);
    }
}
