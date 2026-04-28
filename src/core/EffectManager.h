#ifndef EFFECTMANAGER_H
#define EFFECTMANAGER_H

#include "PhotoEditorEffect.h"
#include <QObject>
#include <QVector>
#include <memory>
#include <vector>

class IGpuEffect;
class IInteractiveEffect;
class ICropSource;

// Observer view over an effect.  EffectManager owns the underlying object;
// callers receive non-owning raw pointers that stay valid for the lifetime
// of the manager.  The interface fields cache the dynamic_cast results
// done once at addEffect() time so per-frame paths can branch without
// re-running RTTI.
struct EffectEntry {
    PhotoEditorEffect*  effect      = nullptr;
    bool                enabled     = true;
    IGpuEffect*         gpu         = nullptr;
    IInteractiveEffect* interactive = nullptr;
    ICropSource*        crop        = nullptr;
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
    // First effect that implements ICropSource, or nullptr.  Cached on
    // addEffect() so per-frame callers don't re-scan + re-dynamic_cast.
    // cropSource() ignores enabled-state (e.g. for pushing image size on
    // load); activeCropSource() returns null when the owning effect is
    // disabled and is the right choice for crop injection / preview bake.
    ICropSource*               cropSource() const;
    ICropSource*               activeCropSource() const;
    void setEnabled(int index, bool enabled);

signals:
    void effectToggled(int index, bool enabled);

private:
    std::vector<std::unique_ptr<PhotoEditorEffect>> m_owners;
    QVector<EffectEntry>                            m_entries;
    ICropSource*                                    m_cropSource = nullptr;
};

#endif // EFFECTMANAGER_H
