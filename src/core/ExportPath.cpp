#include "ExportPath.h"
#include <QDir>
#include <QFile>
#include <QFileInfo>

namespace ExportPath {

QString resolvePattern(const QString& pattern,
                       const QString& sourcePath,
                       int batchIndex,
                       QDate today) {
    const QString name = QFileInfo(sourcePath).completeBaseName();
    const QString n    = QString("%1").arg(batchIndex, 3, 10, QChar('0'));
    const QString date = today.toString("yyyy-MM-dd");

    QString out = pattern;
    out.replace("{name}", name);
    out.replace("{n}",    n);
    out.replace("{date}", date);
    return out;
}

QString chooseDestination(const ExportOptions::Options& opts,
                          const QString& sourcePath,
                          int batchIndex,
                          std::function<bool(const QString&)> exists) {
    if (!exists)
        exists = [](const QString& p) { return QFile::exists(p); };

    const QString stem      = resolvePattern(opts.filenamePattern, sourcePath, batchIndex);
    const QString ext       = ExportOptions::extensionFor(opts.format);
    const QDir    dir(opts.destinationDir);
    const QString candidate = dir.filePath(stem + "." + ext);

    using Policy = ExportOptions::OverwritePolicy;
    switch (opts.onConflict) {
        case Policy::Overwrite:
            return candidate;
        case Policy::Skip:
            return exists(candidate) ? QString() : candidate;
        case Policy::AppendSuffix: {
            if (!exists(candidate)) return candidate;
            // 9999 is a soft cap; if a user has 10k collisions on the same
            // stem they have a different problem to solve.
            for (int i = 1; i <= 9999; ++i) {
                const QString alt =
                    dir.filePath(QString("%1_%2.%3").arg(stem).arg(i).arg(ext));
                if (!exists(alt)) return alt;
            }
            return QString();
        }
    }
    return candidate; // GCOVR_EXCL_LINE — unreachable; satisfies -Wreturn-type
}

} // namespace ExportPath
