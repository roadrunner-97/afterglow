#ifndef STACKPROJECTSTORE_H
#define STACKPROJECTSTORE_H

#include "LongExposureStack.h"

#include <QString>
#include <QVector>

struct StackProject {
    QVector<StackFrame>    frames;
    QString                referencePath;
    StackAggregationConfig aggregation;
};

namespace StackProjectStore {

QString projectDirectory(const QString &folder);
QString manifestPath(const QString &folder);
QString masterPath(const QString &folder);

bool save(const QString &folder, const StackProject &project, QString *error = nullptr);
bool load(const QString &folder, StackProject *project, QString *error = nullptr);

} // namespace StackProjectStore

#endif // STACKPROJECTSTORE_H
