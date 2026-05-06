#ifndef PROOFCACHE_H
#define PROOFCACHE_H

#include <QObject>
#include <QImage>
#include <QString>
#include <QHash>
#include <QList>

// On-disk and in-memory cache for rendered "proofs" — pipeline-processed
// previews that reflect the current sidecar state.
//
// On disk:  <folder>/.afterglow/proofs/<filename>.jpg  (JPEG q=90)
// Freshness: proof.mtime >= sidecar.mtime (or no sidecar exists yet).
//
// In memory: LRU cache of the last 8 decoded QImages for instant round-trips.
class ProofCache : public QObject {
    Q_OBJECT
public:
    explicit ProofCache(QObject* parent = nullptr);

    // Returns the on-disk proof path for a source image.
    static QString proofPath(const QString& imagePath);

    // Returns the expected sidecar (.yml) path for a source image.
    static QString sidecarPath(const QString& imagePath);

    // True if the proof file exists on disk and is at least as new as the
    // sidecar.  If no sidecar exists the proof is always considered fresh.
    bool isProofed(const QString& imagePath) const;

    // Returns the cached proof (from in-memory LRU or disk).
    // Returns a null QImage if the proof does not exist or is stale.
    QImage proof(const QString& imagePath);

    // Write JPEG to disk (quality 90) and populate the in-memory LRU.
    void store(const QString& imagePath, const QImage& proof);

    // Delete the on-disk proof and evict from the in-memory LRU.
    void invalidate(const QString& imagePath);

    // Clear the in-memory LRU.  Disk files are left intact so re-opening the
    // same folder doesn't re-proof already-cached photos.
    void clear();

private:
    static const int MAX_LRU = 8;

    void lruInsert(const QString& key, const QImage& img);
    void lruPromote(const QString& key);
    void lruEvict();

    QList<QString>         m_lruOrder;  // front = most recently used
    QHash<QString, QImage> m_lruCache;
};

#endif // PROOFCACHE_H
