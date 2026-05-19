#include "UndoHistory.h"

UndoHistory::UndoHistory(int capacity, QObject *parent) : QObject(parent), m_capacity(capacity) {}

UndoHistory::Shadow UndoHistory::buildShadow(const QVector<SettingsImporter::EffectSettings> &v) {
    Shadow s;
    s.reserve(v.size());
    for (const auto &e : v) s.insert(e.id, e);
    return s;
}

void UndoHistory::seed(const QVector<SettingsImporter::EffectSettings> &current) {
    const bool prevUndo = canUndo();
    const bool prevRedo = canRedo();
    m_shadow            = buildShadow(current);
    m_entries.clear();
    m_cursor = 0;
    if (prevUndo) emit canUndoChanged(false);
    if (prevRedo) emit canRedoChanged(false);
    emit historyChanged();
}

void UndoHistory::recordFromCurrent(const QVector<SettingsImporter::EffectSettings> &current) {
    if (m_applying || m_shadow.isEmpty()) return;

    const bool prevUndo = canUndo();
    const bool prevRedo = canRedo();

    QVector<Entry> newEntries;
    for (const auto &cur : current) {
        const auto it = m_shadow.constFind(cur.id);
        if (it == m_shadow.constEnd()) continue;
        const auto &old = *it;

        Entry e;
        e.effectId = cur.id;

        if (cur.enabled != old.enabled) e.enabled = {old.enabled, cur.enabled};

        for (auto pit = cur.parameters.cbegin(); pit != cur.parameters.cend(); ++pit) {
            const auto     sit    = old.parameters.constFind(pit.key());
            const QVariant oldVal = (sit != old.parameters.constEnd()) ? sit.value() : QVariant{};
            if (pit.value() != oldVal) e.params.insert(pit.key(), {oldVal, pit.value()});
        }
        for (auto pit = old.parameters.cbegin(); pit != old.parameters.cend(); ++pit) {
            if (!cur.parameters.contains(pit.key())) e.params.insert(pit.key(), {pit.value(), QVariant{}});
        }

        if (!e.empty()) newEntries.append(std::move(e));
    }

    if (newEntries.isEmpty()) return;

    // Truncate redo tail
    if (m_cursor < m_entries.size()) m_entries.resize(m_cursor);

    for (auto &e : newEntries) m_entries.append(std::move(e));
    m_cursor = m_entries.size();

    // Enforce capacity — drop from front, keep cursor in bounds
    while (m_entries.size() > m_capacity) {
        m_entries.removeFirst();
        if (m_cursor > 0) --m_cursor;
    }

    m_shadow = buildShadow(current);

    if (canUndo() != prevUndo) emit canUndoChanged(canUndo());
    if (canRedo() != prevRedo) emit canRedoChanged(canRedo());
    emit historyChanged();
}

bool UndoHistory::canUndo() const {
    return m_cursor > 0;
}
bool UndoHistory::canRedo() const {
    return m_cursor < m_entries.size();
}

std::optional<UndoHistory::Entry> UndoHistory::undo() {
    if (!canUndo()) return std::nullopt;

    const bool prevUndo = canUndo();
    const bool prevRedo = canRedo();

    --m_cursor;
    const Entry result = m_entries[m_cursor];
    updateShadowFrom(result);

    if (canUndo() != prevUndo) emit canUndoChanged(canUndo());
    if (canRedo() != prevRedo) emit canRedoChanged(canRedo());
    emit historyChanged();
    return result;
}

std::optional<UndoHistory::Entry> UndoHistory::redo() {
    if (!canRedo()) return std::nullopt;

    const bool prevUndo = canUndo();
    const bool prevRedo = canRedo();

    const Entry result = m_entries[m_cursor];
    ++m_cursor;
    updateShadowTo(result);

    if (canUndo() != prevUndo) emit canUndoChanged(canUndo());
    if (canRedo() != prevRedo) emit canRedoChanged(canRedo());
    emit historyChanged();
    return result;
}

void UndoHistory::updateShadowFrom(const Entry &e) {
    auto it = m_shadow.find(e.effectId);
    if (it == m_shadow.end()) return;
    if (e.enabled) it->enabled = e.enabled->first;
    for (auto pit = e.params.cbegin(); pit != e.params.cend(); ++pit) {
        if (pit.value().from.isValid()) it->parameters.insert(pit.key(), pit.value().from);
        else it->parameters.remove(pit.key());
    }
}

void UndoHistory::updateShadowTo(const Entry &e) {
    auto it = m_shadow.find(e.effectId);
    if (it == m_shadow.end()) return;
    if (e.enabled) it->enabled = e.enabled->second;
    for (auto pit = e.params.cbegin(); pit != e.params.cend(); ++pit) {
        if (pit.value().to.isValid()) it->parameters.insert(pit.key(), pit.value().to);
        else it->parameters.remove(pit.key());
    }
}

void UndoHistory::setApplying(bool b) {
    m_applying = b;
}
bool UndoHistory::isApplying() const {
    return m_applying;
}

const QVector<UndoHistory::Entry> &UndoHistory::entries() const {
    return m_entries;
}
int UndoHistory::cursor() const {
    return m_cursor;
}

void UndoHistory::load(QVector<Entry> entries, int cursor, QVector<SettingsImporter::EffectSettings> shadow) {
    const bool prevUndo = canUndo();
    const bool prevRedo = canRedo();

    m_entries = std::move(entries);
    m_cursor  = qBound(0, cursor, m_entries.size());
    m_shadow  = buildShadow(shadow);

    if (canUndo() != prevUndo) emit canUndoChanged(canUndo());
    if (canRedo() != prevRedo) emit canRedoChanged(canRedo());
    emit historyChanged();
}
