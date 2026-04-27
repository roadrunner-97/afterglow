#ifndef SETTINGSEXPORTER_H
#define SETTINGSEXPORTER_H

#include <QString>

class EffectManager;

namespace SettingsExporter {

// Serialises the manager's effects (name, enabled, parameter map) to a YAML
// document.  `sourceImagePath` is recorded under `image:` for reference and is
// allowed to be empty (no image loaded yet).
QString toYaml(const EffectManager& mgr, const QString& sourceImagePath = {});

// Convenience: writes the YAML produced by toYaml() to `path`.  Returns true
// on success; on failure, sets `*error` (if non-null) to a human-readable
// message and returns false.
bool writeYaml(const QString& path,
               const EffectManager& mgr,
               const QString& sourceImagePath = {},
               QString* error = nullptr);

} // namespace SettingsExporter

#endif // SETTINGSEXPORTER_H
