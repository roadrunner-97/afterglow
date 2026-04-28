#include "EffectManager.h"
#include "ICropSource.h"
#include "IGpuEffect.h"
#include "IInteractiveEffect.h"
#include <QDebug>

EffectManager::EffectManager(QObject* parent) : QObject(parent) {}

void EffectManager::addEffect(std::unique_ptr<PhotoEditorEffect> effect, bool enabled) {
    PhotoEditorEffect* observer = effect.get();

    EffectEntry entry;
    entry.effect      = observer;
    entry.enabled     = enabled;
    entry.gpu         = dynamic_cast<IGpuEffect*>(observer);
    entry.interactive = dynamic_cast<IInteractiveEffect*>(observer);
    entry.crop        = dynamic_cast<ICropSource*>(observer);

    if (entry.crop && !m_cropSource) m_cropSource = entry.crop;

    m_owners.push_back(std::move(effect));
    m_entries.append(entry);
}

ICropSource* EffectManager::cropSource() const {
    return m_cropSource;
}

ICropSource* EffectManager::activeCropSource() const {
    for (const EffectEntry& e : m_entries)
        if (e.crop && e.enabled) return e.crop;
    return nullptr;
}

const QVector<EffectEntry>& EffectManager::entries() const {
    return m_entries;
}

void EffectManager::setEnabled(int index, bool enabled) {
    if (index < 0 || index >= m_entries.size()) return;
    m_entries[index].enabled = enabled;
    emit effectToggled(index, enabled);
}
