#include "EffectManager.h"
#include <QDebug>

EffectManager::EffectManager(QObject* parent) : QObject(parent) {}

void EffectManager::addEffect(std::unique_ptr<PhotoEditorEffect> effect, bool enabled) {
    PhotoEditorEffect* observer = effect.get();
    m_owners.push_back(std::move(effect));
    m_entries.append({observer, enabled});
}

const QVector<EffectEntry>& EffectManager::entries() const {
    return m_entries;
}

QVector<PhotoEditorEffect*> EffectManager::activeEffects() const {
    QVector<PhotoEditorEffect*> active;
    active.reserve(m_entries.size());
    for (const EffectEntry& e : m_entries)
        if (e.enabled) active.append(e.effect);
    return active;
}

void EffectManager::setEnabled(int index, bool enabled) {
    if (index < 0 || index >= m_entries.size()) return;
    m_entries[index].enabled = enabled;
    emit effectToggled(index, enabled);
}
