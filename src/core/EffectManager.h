#ifndef EFFECTMANAGER_H
#define EFFECTMANAGER_H

#include "PhotoEditorEffect.h"
#include <QObject>
#include <QVector>

struct EffectEntry {
    PhotoEditorEffect* effect = nullptr;
    bool enabled = true;
};

/**
 * @brief Owns all registered effects and tracks their enabled state.
 *
 * Effects are added at startup (in main.cpp) and live for the duration of
 * the application. No dynamic loading — effects are statically linked.
 */
class EffectManager : public QObject {
    Q_OBJECT

public:
    explicit EffectManager(QObject* parent = nullptr);
    ~EffectManager();

    void addEffect(PhotoEditorEffect* effect, bool enabled = true);  // takes ownership
    const QVector<EffectEntry>& entries() const;
    void setEnabled(int index, bool enabled);

signals:
    void effectToggled(int index, bool enabled);

private:
    QVector<EffectEntry> m_entries;
};

#endif // EFFECTMANAGER_H
