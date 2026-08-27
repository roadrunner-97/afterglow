#include "CachePurger.h"

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>

namespace {

bool removeTree(const QString &path, int *filesRemoved, QString *error) {
    const QFileInfo info(path);
    if (!info.exists() && !info.isSymLink()) return true;

    if (info.isSymLink()) {
        if (QFile::remove(path)) {
            ++(*filesRemoved);
            return true;
        }
        *error = QString("Could not remove cache link: %1").arg(path); // GCOVR_EXCL_LINE
        return false;                                                  // GCOVR_EXCL_LINE
    }

    int          count = 0;
    QDirIterator it(path, QDir::Files | QDir::Hidden | QDir::System | QDir::NoDotAndDotDot,
                    QDirIterator::Subdirectories);
    while (it.hasNext()) {
        it.next();
        ++count;
    }

    QDir dir(path);
    if (!dir.removeRecursively()) {
        *error = QString("Could not remove cache directory: %1").arg(path); // GCOVR_EXCL_LINE
        return false;                                                       // GCOVR_EXCL_LINE
    }
    *filesRemoved += count;
    return true;
}

} // namespace

namespace CachePurger {

Result purgePhotoCaches(const QString &folder) {
    Result     result;
    const QDir root(folder);
    if (folder.isEmpty() || !root.exists()) {
        result.error = QString("Photo folder does not exist: %1").arg(folder);
        return result;
    }

    if (!removeTree(root.filePath(".afterglow-thumbs"), &result.filesRemoved, &result.error)) return result;
    if (!removeTree(root.filePath(".afterglow/proofs"), &result.filesRemoved, &result.error)) return result;

    // Remove the generated parent only when it contains nothing else.
    root.rmdir(QStringLiteral(".afterglow"));
    result.success = true;
    return result;
}

} // namespace CachePurger
