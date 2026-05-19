#ifndef HISTORYSERIALIZER_H
#define HISTORYSERIALIZER_H

#include "UndoHistory.h"
#include "SettingsImporter.h"
#include <QString>
#include <QVector>

namespace HistorySerializer {

struct HistoryData {
    int                                       cursor = 0;
    QVector<UndoHistory::Entry>               entries;
    QVector<SettingsImporter::EffectSettings> shadow;
};

QString toYaml(const QVector<UndoHistory::Entry> &entries, int cursor,
               const QVector<SettingsImporter::EffectSettings> &shadow);

bool writeYaml(const QString &path, const QVector<UndoHistory::Entry> &entries, int cursor,
               const QVector<SettingsImporter::EffectSettings> &shadow, QString *error = nullptr);

bool fromYaml(const QString &yaml, HistoryData *out, QString *error = nullptr);
bool readYaml(const QString &path, HistoryData *out, QString *error = nullptr);

} // namespace HistorySerializer

#endif // HISTORYSERIALIZER_H
