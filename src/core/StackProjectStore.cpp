#include "StackProjectStore.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>

namespace {

QJsonObject encodePath(const QString &folder, const QString &path) {
    if (path.isEmpty()) return {};
    const QString absolute = QFileInfo(path).absoluteFilePath();
    const QString relative = QDir(folder).relativeFilePath(absolute);
    const bool external = QDir::isAbsolutePath(relative) || relative == QStringLiteral("..") ||
                          relative.startsWith(QStringLiteral("../"));
    return {{QStringLiteral("path"), external ? absolute : relative}, {QStringLiteral("absolute"), external}};
}

QString decodePath(const QString &folder, const QJsonValue &value) {
    const QJsonObject encoded = value.toObject();
    const QString path = encoded.value(QStringLiteral("path")).toString();
    if (path.isEmpty()) return {};
    return QFileInfo(encoded.value(QStringLiteral("absolute")).toBool() ? path : QDir(folder).filePath(path))
        .absoluteFilePath();
}

void setError(QString *error, const QString &message) {
    if (error) *error = message;
}

} // namespace

namespace StackProjectStore {

QString projectDirectory(const QString &folder) {
    return QDir(folder).filePath(QStringLiteral(".afterglow/stacks/working"));
}

QString manifestPath(const QString &folder) {
    return QDir(projectDirectory(folder)).filePath(QStringLiteral("project.yml"));
}

QString masterPath(const QString &folder) {
    return QDir(projectDirectory(folder)).filePath(QStringLiteral("master.exr"));
}

bool save(const QString &folder, const StackProject &project, QString *error) {
    if (folder.isEmpty() || !QFileInfo(folder).isDir()) {
        setError(error, QStringLiteral("The stack project folder does not exist."));
        return false;
    }
    if (!QDir().mkpath(projectDirectory(folder))) {
        setError(error, QStringLiteral("Could not create the stack project directory."));
        return false;
    }

    QJsonArray frames;
    for (const StackFrame &frame : project.frames) {
        QJsonObject value = encodePath(folder, frame.path);
        value.insert(QStringLiteral("decision"),
                     frame.decision == StackFrameDecision::Include ? QStringLiteral("include")
                                                                    : QStringLiteral("exclude"));
        frames.append(value);
    }
    const QJsonObject aggregation{{QStringLiteral("method"), project.aggregation.methodId},
                                  {QStringLiteral("parameters"),
                                   QJsonObject::fromVariantMap(project.aggregation.parameters)}};
    const QJsonObject root{{QStringLiteral("schema"), 1},
                           {QStringLiteral("frames"), frames},
                           {QStringLiteral("reference"), encodePath(folder, project.referencePath)},
                           {QStringLiteral("aggregation"), aggregation}};

    QSaveFile file(manifestPath(folder));
    if (!file.open(QIODevice::WriteOnly)) {
        setError(error, QStringLiteral("Could not open the stack project manifest for writing."));
        return false;
    }
    const QByteArray data = QJsonDocument(root).toJson(QJsonDocument::Indented);
    // QSaveFile commit/write failure after a successful open requires an
    // external filesystem fault and is not safely inducible in unit tests.
    // GCOVR_EXCL_START
    if (file.write(data) != data.size() || !file.commit()) {
        setError(error, QStringLiteral("Could not save the stack project manifest."));
        return false;
    }
    // GCOVR_EXCL_STOP
    return true;
}

bool load(const QString &folder, StackProject *project, QString *error) {
    if (!project) {
        setError(error, QStringLiteral("No stack project destination was supplied."));
        return false;
    }
    QFile file(manifestPath(folder));
    if (!file.exists()) return false;
    if (!file.open(QIODevice::ReadOnly)) {
        setError(error, QStringLiteral("Could not open the stack project manifest."));
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (document.isNull() || !document.isObject()) {
        setError(error, QStringLiteral("Could not parse the stack project: %1").arg(parseError.errorString()));
        return false;
    }
    const QJsonObject root = document.object();
    if (root.value(QStringLiteral("schema")).toInt() != 1) {
        setError(error, QStringLiteral("The stack project uses an unsupported schema."));
        return false;
    }

    StackProject loaded;
    for (const QJsonValue &value : root.value(QStringLiteral("frames")).toArray()) {
        const QString path = decodePath(folder, value);
        if (path.isEmpty()) continue;
        loaded.frames.append(
            {path, value.toObject().value(QStringLiteral("decision")).toString() == QStringLiteral("exclude")
                       ? StackFrameDecision::Exclude
                       : StackFrameDecision::Include});
    }
    loaded.referencePath = decodePath(folder, root.value(QStringLiteral("reference")));
    const QJsonObject aggregation = root.value(QStringLiteral("aggregation")).toObject();
    loaded.aggregation.methodId = aggregation.value(QStringLiteral("method")).toString(
        QStringLiteral("per-channel-maximum"));
    loaded.aggregation.parameters = aggregation.value(QStringLiteral("parameters")).toObject().toVariantMap();
    *project = std::move(loaded);
    return true;
}

} // namespace StackProjectStore
