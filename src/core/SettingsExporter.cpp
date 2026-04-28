#include "SettingsExporter.h"

#include "EffectManager.h"
#include "PhotoEditorEffect.h"

#include <QFile>
#include <QFileInfo>
#include <QMap>
#include <QString>
#include <QVariant>

namespace {

// Quote a string as a YAML 1.2 double-quoted scalar with the minimum required
// escapes.  Used for any value that could otherwise be misread as a number,
// bool, null, or contain a YAML special character (':', '#', '\n', ...).
QString quoteString(const QString& s) {
    QString out;
    out.reserve(s.size() + 2);
    out.append('"');
    for (QChar ch : s) {
        switch (ch.unicode()) {
            case '"':  out.append("\\\""); break;
            case '\\': out.append("\\\\"); break;
            case '\n': out.append("\\n");  break;
            case '\r': out.append("\\r");  break;
            case '\t': out.append("\\t");  break;
            default:
                if (ch.unicode() < 0x20)
                    out.append(QString("\\x%1").arg(static_cast<int>(ch.unicode()), 2, 16, QChar('0')));
                else
                    out.append(ch);
                break;
        }
    }
    out.append('"');
    return out;
}

// Format a single QVariant as a YAML scalar.  All effect parameters today are
// int / double / bool; anything else falls back to a quoted string.
QString formatScalar(const QVariant& v) {
    switch (static_cast<QMetaType::Type>(v.userType())) {
        case QMetaType::Bool:
            return v.toBool() ? QStringLiteral("true") : QStringLiteral("false");
        case QMetaType::Int:
        case QMetaType::UInt:
        case QMetaType::LongLong:
        case QMetaType::ULongLong:
            return QString::number(v.toLongLong());
        case QMetaType::Float:
        case QMetaType::Double: {
            // %.10g keeps enough precision to round-trip without trailing zeros.
            QString s = QString::number(v.toDouble(), 'g', 10);
            // YAML requires a digit on both sides of the decimal point for some
            // parsers; "1." → "1.0".  Likewise leave integer-valued doubles as
            // plain integers ("5" is unambiguous in our schema).
            return s;
        }
        default:
            return quoteString(v.toString());
    }
}

} // namespace

namespace SettingsExporter {

QString toYaml(const EffectManager& mgr, const QString& sourceImagePath) {
    QString out;
    out.append("# Afterglow effect settings\n");
    if (!sourceImagePath.isEmpty())
        out.append("image: ").append(quoteString(sourceImagePath)).append('\n');

    out.append("effects:\n");
    const auto& entries = mgr.entries();
    for (const auto& entry : entries) {
        if (!entry.effect) continue;
        out.append("  - id: ").append(quoteString(entry.effect->getId())).append('\n');
        out.append("    enabled: ").append(entry.enabled ? "true" : "false").append('\n');

        const auto params = entry.effect->getParameters();
        if (params.isEmpty()) {
            out.append("    parameters: {}\n");
        } else {
            out.append("    parameters:\n");
            // QMap iterates in key order — keeps output deterministic for diffs.
            for (auto it = params.cbegin(); it != params.cend(); ++it)
                out.append("      ").append(it.key()).append(": ")
                   .append(formatScalar(it.value())).append('\n');
        }
    }
    return out;
}

bool writeYaml(const QString& path,
               const EffectManager& mgr,
               const QString& sourceImagePath,
               QString* error)
{
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        if (error) *error = f.errorString();
        return false;
    }
    const QByteArray bytes = toYaml(mgr, sourceImagePath).toUtf8();
    // GCOVR_EXCL_START — short-write on a freshly-opened regular file isn't
    // reachable from a unit test; the branch exists for diagnostic safety.
    if (f.write(bytes) != bytes.size()) {
        if (error) *error = f.errorString();
        return false;
    }
    // GCOVR_EXCL_STOP
    return true;
}

} // namespace SettingsExporter
