#include "EffectManager.h"
#include <QDebug>

EffectManager::EffectManager(QObject* parent) : QObject(parent) {}

EffectManager::~EffectManager() {
    for (const EffectEntry& e : m_entries)
        delete e.effect;
}

void EffectManager::addEffect(PhotoEditorEffect* effect, bool enabled) {
    m_entries.append({effect, enabled});
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
