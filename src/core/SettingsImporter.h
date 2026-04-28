#ifndef SETTINGSIMPORTER_H
#define SETTINGSIMPORTER_H

#include <QMap>
#include <QString>
#include <QVariant>
#include <QVector>

class EffectManager;

namespace SettingsImporter {

struct EffectSettings {
    // `id` is the preferred key for matching against effects (stable,
    // snake-cased) — written by SettingsExporter as of the getId()
    // migration.  `name` is read from older sidecars where the entry
    // was keyed on the user-facing display name; applyToManager falls
    // back to it when no id match is found.
    QString                 id;
    QString                 name;
    bool                    enabled = true;
    QMap<QString, QVariant> parameters;
};

struct Settings {
    QString                  image;
    QVector<EffectSettings>  effects;
};

// Parses the line-oriented YAML subset emitted by SettingsExporter::toYaml.
// Returns true on success; the parser is lenient — unknown keys, missing
// fields, and extra blank/comment lines are accepted silently.
bool fromYaml(const QString& yaml, Settings* out, QString* error = nullptr);

// Convenience: reads `path` and calls fromYaml().
bool readYaml(const QString& path, Settings* out, QString* error = nullptr);

// Walks the manager's effect list, matches by getName() (case-sensitive), and
// for each match: sets the enabled bit and calls applyParameters().  Effects
// not present in `s` are left untouched; entries in `s` with no matching
// effect are skipped.
void applyToManager(const Settings& s, EffectManager& mgr);

} // namespace SettingsImporter

#endif // SETTINGSIMPORTER_H
