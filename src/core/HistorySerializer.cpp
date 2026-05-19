#include "HistorySerializer.h"

#include <QFile>
#include <QString>
#include <QStringList>
#include <QVariant>
#include <climits>

namespace {

// ─── Scalar serialisation (mirrors SettingsExporter) ───────────────────────

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
                if (ch.unicode() < 0x20) // GCOVR_EXCL_LINE
                    out.append(QString("\\x%1").arg( // GCOVR_EXCL_LINE
                        static_cast<int>(ch.unicode()), 2, 16, QChar('0'))); // GCOVR_EXCL_LINE
                else
                    out.append(ch);
                break;
        }
    }
    out.append('"');
    return out;
}

QString formatScalar(const QVariant& v) {
    if (!v.isValid()) return QStringLiteral("null");
    switch (static_cast<QMetaType::Type>(v.userType())) {
        case QMetaType::Bool:
            return v.toBool() ? QStringLiteral("true") : QStringLiteral("false");
        case QMetaType::Int:
        case QMetaType::UInt:
        case QMetaType::LongLong:
        case QMetaType::ULongLong:
            return QString::number(v.toLongLong());
        case QMetaType::Float:
        case QMetaType::Double:
            return QString::number(v.toDouble(), 'g', 10);
        default:
            return quoteString(v.toString());
    }
}

// ─── Scalar parsing (mirrors SettingsImporter) ─────────────────────────────

QString unquoteStr(const QString& token) {
    if (token.size() < 2 || !token.startsWith('"') || !token.endsWith('"'))
        return token; // GCOVR_EXCL_LINE
    const QString inner = token.mid(1, token.size() - 2);
    QString out;
    out.reserve(inner.size());
    for (int i = 0; i < inner.size(); ++i) {
        const QChar c = inner[i];
        if (c != '\\' || i + 1 >= inner.size()) { out.append(c); continue; }
        const QChar n = inner[++i];
        switch (n.unicode()) {
            case '"':  out.append('"');  break;
            case '\\': out.append('\\'); break;
            case 'n':  out.append('\n'); break;
            case 'r':  out.append('\r'); break;
            case 't':  out.append('\t'); break;
            // GCOVR_EXCL_START — \xNN sequences are only written for ASCII
            // control characters; quoteString exclusion matches here too.
            case 'x': {
                if (i + 2 < inner.size()) {
                    bool ok = false;
                    const int code = inner.mid(i + 1, 2).toInt(&ok, 16);
                    if (ok) {
                        out.append(QChar(static_cast<char16_t>(code)));
                        i += 2;
                        break;
                    }
                }
                out.append(n);
                break;
            }
            // GCOVR_EXCL_STOP
            default: out.append(n); break; // GCOVR_EXCL_LINE
        }
    }
    return out;
}

QVariant parseScalarV(const QString& s) {
    const QString t = s.trimmed();
    if (t == QStringLiteral("null"))  return QVariant{};
    if (t == QStringLiteral("true"))  return true;
    if (t == QStringLiteral("false")) return false;
    if (t.startsWith('"'))            return unquoteStr(t);
    bool ok = false;
    if (!t.contains('.') && !t.contains('e') && !t.contains('E')) {
        const long long ll = t.toLongLong(&ok);
        if (ok) {
            if (ll >= INT_MIN && ll <= INT_MAX)
                return static_cast<int>(ll);
            return ll;
        }
    }
    const double d = t.toDouble(&ok);
    if (ok) return d;
    return t; // GCOVR_EXCL_LINE
}

bool splitKV(const QString& s, QString* k, QString* v) {
    const int colon = s.indexOf(':');
    if (colon < 0) return false;
    *k = s.left(colon).trimmed();
    *v = s.mid(colon + 1).trimmed();
    return true;
}

int leadingSpaces(const QString& line) {
    int i = 0;
    while (i < line.size() && line[i] == ' ') ++i;
    return i;
}

// Parse "{ from: X, to: Y }" — values must be non-quoted scalars or "null".
bool parseFromTo(const QString& val, QVariant* fromOut, QVariant* toOut) {
    QString t = val.trimmed();
    if (!t.startsWith('{') || !t.endsWith('}')) return false;
    t = t.mid(1, t.size() - 2).trimmed();

    QVariant fromV, toV;
    bool gotFrom = false, gotTo = false;

    for (const QString& raw : t.split(',')) {
        const QString part = raw.trimmed();
        QString k, v;
        if (!splitKV(part, &k, &v)) return false;
        if (k == QStringLiteral("from")) { fromV = parseScalarV(v); gotFrom = true; }
        else if (k == QStringLiteral("to")) { toV = parseScalarV(v); gotTo = true; }
    }
    if (!gotFrom || !gotTo) return false;
    *fromOut = fromV;
    *toOut   = toV;
    return true;
}

} // namespace

namespace HistorySerializer {

QString toYaml(const QVector<UndoHistory::Entry>& entries,
               int cursor,
               const QVector<SettingsImporter::EffectSettings>& shadow)
{
    QString out;
    out.append("# Afterglow undo history\n");
    out.append("cursor: ").append(QString::number(cursor)).append('\n');

    out.append("shadow:\n");
    for (const auto& s : shadow) {
        out.append("  - id: ").append(quoteString(s.id)).append('\n');
        out.append("    enabled: ").append(s.enabled ? "true" : "false").append('\n');
        if (s.parameters.isEmpty()) {
            out.append("    parameters: {}\n");
        } else {
            out.append("    parameters:\n");
            for (auto it = s.parameters.cbegin(); it != s.parameters.cend(); ++it)
                out.append("      ").append(it.key()).append(": ")
                   .append(formatScalar(it.value())).append('\n');
        }
    }

    out.append("entries:\n");
    for (const auto& e : entries) {
        out.append("  - effect: ").append(quoteString(e.effectId)).append('\n');
        if (e.enabled) {
            out.append("    enabled: { from: ")
               .append(e.enabled->first ? "true" : "false")
               .append(", to: ")
               .append(e.enabled->second ? "true" : "false")
               .append(" }\n");
        }
        if (!e.params.isEmpty()) {
            out.append("    params:\n");
            for (auto pit = e.params.cbegin(); pit != e.params.cend(); ++pit)
                out.append("      ").append(pit.key())
                   .append(": { from: ").append(formatScalar(pit.value().from))
                   .append(", to: ").append(formatScalar(pit.value().to))
                   .append(" }\n");
        }
    }
    return out;
}

bool writeYaml(const QString& path,
               const QVector<UndoHistory::Entry>& entries,
               int cursor,
               const QVector<SettingsImporter::EffectSettings>& shadow,
               QString* error)
{
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        if (error) *error = f.errorString();
        return false;
    }
    const QByteArray bytes = toYaml(entries, cursor, shadow).toUtf8();
    // GCOVR_EXCL_START
    if (f.write(bytes) != bytes.size()) {
        if (error) *error = f.errorString();
        return false;
    }
    // GCOVR_EXCL_STOP
    return true;
}

bool fromYaml(const QString& yaml, HistoryData* out, QString* error) {
    out->cursor = 0;
    out->entries.clear();
    out->shadow.clear();
    if (error) error->clear();

    enum class Section { None, Shadow, Entries };
    Section section  = Section::None;
    SettingsImporter::EffectSettings* curShadow = nullptr;
    UndoHistory::Entry*               curEntry  = nullptr;
    bool inParams = false;

    const QStringList lines = yaml.split('\n');
    for (const QString& raw : lines) {
        QString line = raw;
        while (!line.isEmpty() && line.back().isSpace()) line.chop(1);
        if (line.isEmpty()) continue;

        const int indent    = leadingSpaces(line);
        const QString rest  = line.mid(indent);
        if (rest.startsWith('#')) continue;

        if (indent == 0) {
            inParams   = false;
            curShadow  = nullptr;
            curEntry   = nullptr;
            QString k, v;
            if (!splitKV(rest, &k, &v)) continue;
            if (k == QStringLiteral("cursor")) {
                bool ok = false;
                const int c = v.toInt(&ok);
                if (ok) out->cursor = c;
            } else if (k == QStringLiteral("shadow")) {
                section = Section::Shadow;
            } else if (k == QStringLiteral("entries")) {
                section = Section::Entries;
            }

        } else if (indent == 2) {
            inParams = false;
            if (!rest.startsWith(QStringLiteral("- "))) continue;
            const QString afterDash = rest.mid(2).trimmed();
            QString k, v;
            if (!splitKV(afterDash, &k, &v)) continue;

            if (section == Section::Shadow) {
                SettingsImporter::EffectSettings entry;
                if (k == QStringLiteral("id")) entry.id = parseScalarV(v).toString();
                out->shadow.append(entry);
                curShadow = &out->shadow.last();
                curEntry  = nullptr;
            } else if (section == Section::Entries) {
                UndoHistory::Entry entry;
                if (k == QStringLiteral("effect")) entry.effectId = parseScalarV(v).toString();
                out->entries.append(entry);
                curEntry  = &out->entries.last();
                curShadow = nullptr;
            }

        } else if (indent == 4) {
            QString k, v;
            if (!splitKV(rest, &k, &v)) continue;

            if (curShadow) {
                if      (k == QStringLiteral("enabled"))    curShadow->enabled = parseScalarV(v).toBool();
                else if (k == QStringLiteral("id"))         curShadow->id      = parseScalarV(v).toString();
                else if (k == QStringLiteral("parameters")) inParams = true;
            } else if (curEntry) {
                if (k == QStringLiteral("enabled")) {
                    QVariant from, to;
                    if (parseFromTo(v, &from, &to))
                        curEntry->enabled = {from.toBool(), to.toBool()};
                } else if (k == QStringLiteral("params")) {
                    inParams = true;
                }
            }

        } else if (indent == 6) {
            if (!inParams) continue;
            QString k, v;
            if (!splitKV(rest, &k, &v)) continue;

            if (curShadow) {
                curShadow->parameters.insert(k, parseScalarV(v));
            } else if (curEntry) {
                QVariant from, to;
                if (parseFromTo(v, &from, &to))
                    curEntry->params.insert(k, {from, to});
            }
        }
    }

    return true;
}

bool readYaml(const QString& path, HistoryData* out, QString* error) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) *error = f.errorString();
        return false;
    }
    return fromYaml(QString::fromUtf8(f.readAll()), out, error);
}

} // namespace HistorySerializer
