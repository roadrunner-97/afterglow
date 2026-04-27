#include "SettingsImporter.h"

#include "EffectManager.h"
#include "PhotoEditorEffect.h"

#include <QFile>
#include <QStringList>

#include <climits>

namespace {

QString unquote(const QString& token) {
    if (token.size() < 2 || !token.startsWith('"') || !token.endsWith('"'))
        return token;
    const QString inner = token.mid(1, token.size() - 2);
    QString out;
    out.reserve(inner.size());
    for (int i = 0; i < inner.size(); ++i) {
        const QChar c = inner[i];
        if (c != '\\' || i + 1 >= inner.size()) {
            out.append(c);
            continue;
        }
        const QChar n = inner[++i];
        switch (n.unicode()) {
            case '"':  out.append('"');  break;
            case '\\': out.append('\\'); break;
            case 'n':  out.append('\n'); break;
            case 'r':  out.append('\r'); break;
            case 't':  out.append('\t'); break;
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
            default: out.append(n); break;
        }
    }
    return out;
}

QVariant parseScalar(const QString& s) {
    const QString t = s.trimmed();
    if (t == QStringLiteral("true"))  return true;
    if (t == QStringLiteral("false")) return false;
    if (t.startsWith('"')) return unquote(t);

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
    return t; // unrecognised — keep as raw string
}

int leadingSpaces(const QString& line) {
    int i = 0;
    while (i < line.size() && line[i] == ' ') ++i;
    return i;
}

bool splitKeyValue(const QString& s, QString* k, QString* v) {
    const int colon = s.indexOf(':');
    if (colon < 0) return false;
    *k = s.left(colon).trimmed();
    *v = s.mid(colon + 1).trimmed();
    return true;
}

} // namespace

namespace SettingsImporter {

bool fromYaml(const QString& yaml, Settings* out, QString* /*error*/) {
    out->image.clear();
    out->effects.clear();

    EffectSettings* current = nullptr;
    const QStringList lines = yaml.split('\n');

    for (const QString& raw : lines) {
        QString line = raw;
        while (!line.isEmpty() && line.back().isSpace()) line.chop(1);
        if (line.isEmpty()) continue;

        const int indent = leadingSpaces(line);
        const QString rest = line.mid(indent);
        if (rest.startsWith('#')) continue;

        QString k, v;
        if (indent == 0) {
            if (!splitKeyValue(rest, &k, &v)) continue;
            if (k == QStringLiteral("image"))
                out->image = parseScalar(v).toString();
            // "effects:" header is implicit — its children appear at indent 2+
        } else if (indent == 2) {
            if (!rest.startsWith(QStringLiteral("- "))) continue;
            const QString afterDash = rest.mid(2).trimmed();
            if (!splitKeyValue(afterDash, &k, &v)) continue;
            EffectSettings entry;
            if (k == QStringLiteral("name"))
                entry.name = parseScalar(v).toString();
            out->effects.append(entry);
            current = &out->effects.last();
        } else if (indent == 4) {
            if (!current) continue;
            if (!splitKeyValue(rest, &k, &v)) continue;
            if (k == QStringLiteral("enabled"))
                current->enabled = parseScalar(v).toBool();
            // "parameters:" is implicit — child entries follow at indent 6
        } else if (indent == 6) {
            if (!current) continue;
            if (!splitKeyValue(rest, &k, &v)) continue;
            current->parameters[k] = parseScalar(v);
        }
    }

    return true;
}

bool readYaml(const QString& path, Settings* out, QString* error) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) *error = f.errorString();
        return false;
    }
    const QString yaml = QString::fromUtf8(f.readAll());
    return fromYaml(yaml, out, error);
}

void applyToManager(const Settings& s, EffectManager& mgr) {
    const auto& entries = mgr.entries();
    for (const auto& want : s.effects) {
        for (int i = 0; i < entries.size(); ++i) {
            PhotoEditorEffect* effect = entries[i].effect;
            if (!effect || effect->getName() != want.name) continue;
            mgr.setEnabled(i, want.enabled);
            effect->applyParameters(want.parameters);
            break;
        }
    }
}

} // namespace SettingsImporter
