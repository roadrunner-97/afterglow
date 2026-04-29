#ifndef EXPORTPATH_H
#define EXPORTPATH_H

#include <QDate>
#include <QString>
#include <functional>
#include "ExportOptions.h"

namespace ExportPath {

// Substitutes brace tokens in `pattern`:
//   {name} → completeBaseName of sourcePath (e.g. "IMG_1234" from /a/b/IMG_1234.cr2)
//   {n}    → batchIndex zero-padded to 3 digits (1 → "001")
//   {date} → today as ISO yyyy-MM-dd
// `today` is injected so tests are deterministic; defaults to the current date.
QString resolvePattern(const QString& pattern,
                       const QString& sourcePath,
                       int batchIndex,
                       QDate today = QDate::currentDate());

// Builds the final destination path: destinationDir / resolvePattern(...) +
// extensionFor(format).  Then applies opts.onConflict:
//   Overwrite    → returns the candidate as-is
//   Skip         → returns "" if the candidate exists
//   AppendSuffix → walks _1, _2, ... until a free name is found (caps at 9999)
//
// `exists` is the existence predicate; defaults to QFile::exists.  Tests
// inject a stub so they don't touch the real filesystem.
QString chooseDestination(const ExportOptions::Options& opts,
                          const QString& sourcePath,
                          int batchIndex,
                          std::function<bool(const QString&)> exists = {});

} // namespace ExportPath

#endif // EXPORTPATH_H
