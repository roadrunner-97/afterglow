#ifndef EFFECTMANAGER_H
#define EFFECTMANAGER_H

#include "PhotoEditorEffect.h"
#include <QObject>
#include <QVector>
#include <memory>
#include <vector>

// Observer view over an effect.  EffectManager owns the underlying object;
// callers receive a non-owning raw pointer that stays valid for the
// lifetime of the manager.
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

    void addEffect(std::unique_ptr<PhotoEditorEffect> effect, bool enabled = true);
    const QVector<EffectEntry>& entries() const;
    QVector<PhotoEditorEffect*> activeEffects() const;
    void setEnabled(int index, bool enabled);

signals:
    void effectToggled(int index, bool enabled);

private:
    std::vector<std::unique_ptr<PhotoEditorEffect>> m_owners;
    QVector<EffectEntry>                            m_entries;
};

#endif // EFFECTMANAGER_H
