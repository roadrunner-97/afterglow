#include "EffectManager.h"
#include "ICropSource.h"
#include "IGpuEffect.h"
#include "IInteractiveEffect.h"
#include <QDebug>
#include <QHash>
#include <QSet>

EffectManager::EffectManager(QObject *parent) : QObject(parent) {}

void EffectManager::addEffect(std::unique_ptr<PhotoEditorEffect> effect, bool enabled) {
    PhotoEditorEffect *observer = effect.get();

    EffectEntry entry;
    entry.effect      = observer;
    entry.enabled     = enabled;
    entry.gpu         = dynamic_cast<IGpuEffect *>(observer);
    entry.interactive = dynamic_cast<IInteractiveEffect *>(observer);
    entry.crop        = dynamic_cast<ICropSource *>(observer);

    if (entry.crop && !m_cropSource) m_cropSource = entry.crop;

    m_owners.push_back(std::move(effect));
    m_entries.append(entry);
}

ICropSource *EffectManager::cropSource() const {
    return m_cropSource;
}

ICropSource *EffectManager::activeCropSource() const {
    for (const EffectEntry &e : m_entries)
        if (e.crop && e.enabled) return e.crop;
    return nullptr;
}

const QVector<EffectEntry> &EffectManager::entries() const {
    return m_entries;
}

void EffectManager::setProcessingParameterOverride(const QString &effectId, const QMap<QString, QVariant> &parameters) {
    m_parameterOverrides.insert(effectId, parameters);
}

void EffectManager::clearProcessingParameterOverride(const QString &effectId) {
    m_parameterOverrides.remove(effectId);
}

QMap<QString, QVariant> EffectManager::effectiveParameters(const EffectEntry &entry) const {
    if (!entry.effect) return {};
    const auto override = m_parameterOverrides.constFind(entry.effect->getId());
    return override == m_parameterOverrides.constEnd() ? entry.effect->getParameters() : override.value();
}

void EffectManager::setEnabled(int index, bool enabled) {
    if (index < 0 || index >= m_entries.size()) return;
    m_entries[index].enabled = enabled;
    emit effectToggled(index, enabled);
}

void EffectManager::configureEffects(const QVector<QPair<QString, bool>> &effects) {
    QHash<QString, EffectEntry> byId;
    byId.reserve(m_entries.size());
    for (const EffectEntry &entry : m_entries)
        if (entry.effect) byId.insert(entry.effect->getId(), entry);

    QVector<EffectEntry> reordered;
    reordered.reserve(m_entries.size());
    QSet<QString> seen;
    for (const auto &requested : effects) {
        if (seen.contains(requested.first)) continue;
        const auto it = byId.constFind(requested.first);
        if (it == byId.constEnd()) continue;
        EffectEntry entry = it.value();
        entry.enabled     = requested.second;
        reordered.append(entry);
        seen.insert(requested.first);
    }
    for (const EffectEntry &entry : m_entries) {
        if (!entry.effect || seen.contains(entry.effect->getId())) continue;
        reordered.append(entry);
    }

    bool orderChanged = reordered.size() == m_entries.size();
    if (orderChanged) {
        orderChanged = false;
        for (int i = 0; i < reordered.size(); ++i) {
            if (reordered[i].effect != m_entries[i].effect) {
                orderChanged = true;
                break;
            }
        }
    }

    QHash<PhotoEditorEffect *, bool> oldEnabled;
    for (const EffectEntry &entry : m_entries) oldEnabled.insert(entry.effect, entry.enabled);
    m_entries = std::move(reordered);

    if (orderChanged) emit effectsReordered();
    for (int i = 0; i < m_entries.size(); ++i) {
        if (oldEnabled.value(m_entries[i].effect) != m_entries[i].enabled) emit effectToggled(i, m_entries[i].enabled);
    }
}
