#ifndef LOCALADJUSTMENT_H
#define LOCALADJUSTMENT_H

#include "LinearGradientMask.h"

#include <QString>
#include <QVector>

struct LocalAdjustment {
    QString            id;
    QString            name;
    bool               enabled    = true;
    double             exposureEv = 0.0;
    LinearGradientMask mask;
};

// Ordered, processing-independent collection of local adjustments. IDs are
// stable across sidecar round trips and are the key used by future history
// entries; display names are deliberately not identifiers.
class LocalAdjustmentStack {
public:
    const QVector<LocalAdjustment> &adjustments() const;
    bool                            isEmpty() const;

    QString addLinearGradient(const LinearGradientMask &mask, const QString &name = {});
    bool    appendRestored(LocalAdjustment adjustment);
    bool    remove(const QString &id);
    bool    move(const QString &id, int destinationIndex);
    void    clear();

    LocalAdjustment       *find(const QString &id);
    const LocalAdjustment *find(const QString &id) const;

private:
    QString uniqueDefaultName() const;

    QVector<LocalAdjustment> m_adjustments;
};

#endif // LOCALADJUSTMENT_H
