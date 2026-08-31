#include "LocalAdjustment.h"

#include <QUuid>
#include <algorithm>

const QVector<LocalAdjustment> &LocalAdjustmentStack::adjustments() const {
    return m_adjustments;
}

bool LocalAdjustmentStack::isEmpty() const {
    return m_adjustments.isEmpty();
}

QString LocalAdjustmentStack::uniqueDefaultName() const {
    int candidate = 1;
    while (true) {
        const QString name = QStringLiteral("Linear Gradient %1").arg(candidate);
        const bool    used = std::any_of(m_adjustments.cbegin(), m_adjustments.cend(),
                                         [&name](const LocalAdjustment &item) { return item.name == name; });
        if (!used) return name;
        ++candidate;
    }
}

QString LocalAdjustmentStack::addLinearGradient(const LinearGradientMask &mask, const QString &name) {
    LocalAdjustment adjustment;
    adjustment.id   = QUuid::createUuid().toString(QUuid::WithoutBraces);
    adjustment.name = name.isEmpty() ? uniqueDefaultName() : name;
    adjustment.mask = mask;
    m_adjustments.append(adjustment);
    return adjustment.id;
}

bool LocalAdjustmentStack::appendRestored(LocalAdjustment adjustment) {
    if (adjustment.id.isEmpty() || find(adjustment.id)) return false;
    if (adjustment.name.isEmpty()) adjustment.name = uniqueDefaultName();
    m_adjustments.append(std::move(adjustment));
    return true;
}

bool LocalAdjustmentStack::remove(const QString &id) {
    for (qsizetype i = 0; i < m_adjustments.size(); ++i) {
        if (m_adjustments[i].id != id) continue;
        m_adjustments.removeAt(i);
        return true;
    }
    return false;
}

bool LocalAdjustmentStack::move(const QString &id, int destinationIndex) {
    if (destinationIndex < 0 || destinationIndex >= m_adjustments.size()) return false;
    for (qsizetype i = 0; i < m_adjustments.size(); ++i) {
        if (m_adjustments[i].id != id) continue;
        if (i == destinationIndex) return true;
        m_adjustments.move(i, destinationIndex);
        return true;
    }
    return false;
}

void LocalAdjustmentStack::clear() {
    m_adjustments.clear();
}

LocalAdjustment *LocalAdjustmentStack::find(const QString &id) {
    for (LocalAdjustment &adjustment : m_adjustments)
        if (adjustment.id == id) return &adjustment;
    return nullptr;
}

const LocalAdjustment *LocalAdjustmentStack::find(const QString &id) const {
    for (const LocalAdjustment &adjustment : m_adjustments)
        if (adjustment.id == id) return &adjustment;
    return nullptr;
}
