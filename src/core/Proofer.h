#ifndef PROOFER_H
#define PROOFER_H

#include <QObject>
#include <QImage>
#include <QString>
#include <QStringList>
#include <QHash>
#include <cstdint>
#include <memory>
#include "EffectManager.h"
#include "GpuPipeline.h"
#include "SettingsImporter.h"

class ProofCache;

// Background worker that generates pipeline-rendered proofs for photos in the
// current folder.  Runs one job at a time via QtConcurrent so the proofer
// never saturates the GPU while Develop is active.
//
// All public methods must be called from the main (GUI) thread.
class Proofer : public QObject {
    Q_OBJECT
public:
    // effects: Proofer's own EffectManager with separate effect instances —
    //          never shared with the Develop pipeline.
    // defaults: parameter snapshot at application start (from snapshotDefaults).
    // cache: not owned; must outlive this Proofer.
    Proofer(std::unique_ptr<EffectManager> effects, SettingsImporter::Settings defaults, ProofCache *cache,
            QObject *parent = nullptr);
    ~Proofer() override;

    // Replace the proof queue.  Caller should pre-filter already-proofed paths.
    void setQueue(QStringList paths);

    // Move path to the front of the queue.  If not currently in the queue
    // (e.g. photo was proofed earlier but sidecar changed since), it is added.
    // No-op if already at the head.
    void promote(const QString &path);

    // Invalidate any in-flight result for path and queue a fresh proof.
    void refresh(const QString &path);
    void setDefaults(SettingsImporter::Settings defaults);

    // Stop dispatching new jobs.  Any in-progress job runs to completion.
    void pause();

    // Allow dispatching; immediately starts the next job if the queue is
    // non-empty and no job is currently running.
    void resume();

    // Clear the queue. The current GPU job may finish, but its result is
    // discarded through the queue generation check.
    void clear();

    // Number of photos waiting to be proofed (not counting the current job).
    // Exposed for unit-test inspection.
    int pendingCount() const {
        return static_cast<int>(m_queue.size());
    }

signals:
    void proofStarted(QString path);
    void proofFinished(QString path, QImage proof);
    void proofFailed(QString path, QString error);

private:
    void dispatchNext();

    QStringList              m_queue;
    bool                     m_paused          = false;
    bool                     m_busy            = false;
    uint64_t                 m_queueGeneration = 0;
    QHash<QString, uint64_t> m_pathGenerations;

    std::unique_ptr<EffectManager> m_effects;
    SettingsImporter::Settings     m_defaults;
    ProofCache                    *m_cache; // not owned
    std::shared_ptr<GpuPipeline>   m_pipeline;
};

#endif // PROOFER_H
