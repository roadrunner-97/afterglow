#include "SettingsImporter.h"

#include "EffectManager.h"
#include "PhotoEditorEffect.h"

#include <QFile>
#include <QHash>
#include <QSignalBlocker>
#include <QStringList>

#include <climits>

namespace {

QString unquote(const QString &token) {
    if (token.size() < 2 || !token.startsWith('"') || !token.endsWith('"')) return token;
    const QString inner = token.mid(1, token.size() - 2);
    QString       out;
    out.reserve(inner.size());
    for (int i = 0; i < inner.size(); ++i) {
        const QChar c = inner[i];
        if (c != '\\' || i + 1 >= inner.size()) {
            out.append(c);
            continue;
        }
        const QChar n = inner[++i];
        switch (n.unicode()) {
        case '"':
            out.append('"');
            break;
        case '\\':
            out.append('\\');
            break;
        case 'n':
            out.append('\n');
            break;
        case 'r':
            out.append('\r');
            break;
        case 't':
            out.append('\t');
            break;
        case 'x': {
            if (i + 2 < inner.size()) {
                bool      ok   = false;
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
        default:
            out.append(n);
            break;
        }
    }
    return out;
}

QVariant parseScalar(const QString &s) {
    const QString t = s.trimmed();
    if (t == QStringLiteral("true")) return true;
    if (t == QStringLiteral("false")) return false;
    if (t.startsWith('"')) return unquote(t);

    bool ok = false;
    if (!t.contains('.') && !t.contains('e') && !t.contains('E')) {
        const long long ll = t.toLongLong(&ok);
        if (ok) {
            if (ll >= INT_MIN && ll <= INT_MAX) return static_cast<int>(ll);
            return ll;
        }
    }
    const double d = t.toDouble(&ok);
    if (ok) return d;
    return t; // unrecognised — keep as raw string
}

int leadingSpaces(const QString &line) {
    int i = 0;
    while (i < line.size() && line[i] == ' ') ++i;
    return i;
}

bool splitKeyValue(const QString &s, QString *k, QString *v) {
    const qsizetype colon = s.indexOf(':');
    if (colon < 0) return false;
    *k = s.left(colon).trimmed();
    *v = s.mid(colon + 1).trimmed();
    return true;
}

} // namespace

namespace SettingsImporter {

bool fromYaml(const QString &yaml, Settings *out, QString *error) {
    out->image.clear();
    out->effects.clear();
    out->localAdjustments.clear();
    if (error) error->clear();

    enum class Section { None, Effects, LocalAdjustments };
    Section           section      = Section::None;
    EffectSettings   *current      = nullptr;
    LocalAdjustment  *currentLocal = nullptr;
    const QStringList lines        = yaml.split('\n');

    int lineNo = 0;
    for (const QString &raw : lines) {
        ++lineNo;
        QString line = raw;
        while (!line.isEmpty() && line.back().isSpace()) line.chop(1);
        if (line.isEmpty()) continue;

        // Tabs in leading whitespace silently confused the indent-based
        // dispatch below, producing entries with no name and no params.
        // Reject up front with a concrete diagnostic.
        for (QChar c : line) {
            if (c == '\t') {
                if (error)
                    *error = QString("line %1: tabs are not allowed in leading whitespace; use spaces").arg(lineNo);
                return false;
            }
            if (c != ' ') break;
        }

        const int     indent = leadingSpaces(line);
        const QString rest   = line.mid(indent);
        if (rest.startsWith('#')) continue;

        QString k, v;
        if (indent == 0) {
            if (!splitKeyValue(rest, &k, &v)) continue;
            if (k == QStringLiteral("image")) out->image = parseScalar(v).toString();
            else if (k == QStringLiteral("effects")) section = Section::Effects;
            else if (k == QStringLiteral("local_adjustments")) section = Section::LocalAdjustments;
        } else if (indent == 2) {
            if (!rest.startsWith(QStringLiteral("- "))) continue;
            const QString afterDash = rest.mid(2).trimmed();
            if (!splitKeyValue(afterDash, &k, &v)) continue;
            if (section == Section::LocalAdjustments) {
                LocalAdjustment entry;
                if (k == QStringLiteral("id")) entry.id = parseScalar(v).toString();
                out->localAdjustments.append(entry);
                currentLocal = &out->localAdjustments.last();
                current      = nullptr;
            } else {
                EffectSettings entry;
                if (k == QStringLiteral("id")) entry.id = parseScalar(v).toString();
                else if (k == QStringLiteral("name")) entry.name = parseScalar(v).toString();
                out->effects.append(entry);
                current      = &out->effects.last();
                currentLocal = nullptr;
            }
        } else if (indent == 4) {
            if (!splitKeyValue(rest, &k, &v)) continue;
            if (section == Section::LocalAdjustments && currentLocal) {
                const QVariant value = parseScalar(v);
                if (k == QStringLiteral("name")) currentLocal->name = value.toString();
                else if (k == QStringLiteral("enabled")) currentLocal->enabled = value.toBool();
                else if (k == QStringLiteral("exposure_ev")) currentLocal->exposureEv = value.toDouble();
                else if (k == QStringLiteral("inverted")) currentLocal->mask.setInverted(value.toBool());
                else if (k == QStringLiteral("center_x"))
                    currentLocal->mask.setCenter({value.toDouble(), currentLocal->mask.center().y()});
                else if (k == QStringLiteral("center_y"))
                    currentLocal->mask.setCenter({currentLocal->mask.center().x(), value.toDouble()});
                else if (k == QStringLiteral("direction_x"))
                    currentLocal->mask.setDirection({value.toDouble(), currentLocal->mask.direction().y()});
                else if (k == QStringLiteral("direction_y"))
                    currentLocal->mask.setDirection({currentLocal->mask.direction().x(), value.toDouble()});
                else if (k == QStringLiteral("feather_half_width"))
                    currentLocal->mask.setFeatherHalfWidth(value.toDouble());
            } else if (current) {
                if (k == QStringLiteral("enabled")) current->enabled = parseScalar(v).toBool();
                else if (k == QStringLiteral("id")) current->id = parseScalar(v).toString();
                else if (k == QStringLiteral("name")) current->name = parseScalar(v).toString();
            }
        } else if (indent == 6) {
            if (!current) continue;
            if (!splitKeyValue(rest, &k, &v)) continue;
            current->parameters[k] = parseScalar(v);
        }
    }

    return true;
}

bool readYaml(const QString &path, Settings *out, QString *error) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) *error = f.errorString();
        return false;
    }
    const QString yaml = QString::fromUtf8(f.readAll());
    return fromYaml(yaml, out, error);
}

void applyToManager(const Settings &s, EffectManager &mgr) {
    const auto &initialEntries = mgr.entries();

    // Two parallel hashes: prefer matching by stable id, fall back to
    // display name for older sidecars saved before the id migration.
    QHash<QString, int> indexById;
    QHash<QString, int> indexByName;
    indexById.reserve(initialEntries.size());
    indexByName.reserve(initialEntries.size());
    for (int i = 0; i < initialEntries.size(); ++i) {
        if (!initialEntries[i].effect) continue;
        indexById.insert(initialEntries[i].effect->getId(), i);
        indexByName.insert(initialEntries[i].effect->getName(), i);
    }

    QVector<QPair<QString, bool>> configuration;
    configuration.reserve(s.effects.size());
    for (const auto &want : s.effects) {
        int i = want.id.isEmpty() ? -1 : indexById.value(want.id, -1);
        if (i < 0 && !want.name.isEmpty()) i = indexByName.value(want.name, -1);
        if (i >= 0) configuration.append({initialEntries[i].effect->getId(), want.enabled});
    }
    mgr.configureEffects(configuration);

    const auto &entries = mgr.entries();
    indexById.clear();
    indexByName.clear();
    for (int i = 0; i < entries.size(); ++i) {
        if (!entries[i].effect) continue;
        indexById.insert(entries[i].effect->getId(), i);
        indexByName.insert(entries[i].effect->getName(), i);
    }

    // Block parametersChanged on each effect for the duration of the apply
    // pass.  Without this, every applyParameters() call would queue a full
    // pipeline reprocess; the caller is expected to fire one definitive
    // reprocess after this returns.
    for (const auto &want : s.effects) {
        int i = -1;
        if (!want.id.isEmpty()) {
            const auto it = indexById.constFind(want.id);
            if (it != indexById.constEnd()) i = it.value();
        }
        if (i < 0 && !want.name.isEmpty()) {
            const auto it = indexByName.constFind(want.name);
            if (it != indexByName.constEnd()) i = it.value();
        }
        if (i < 0) continue;
        PhotoEditorEffect *effect = entries[i].effect;
        QSignalBlocker     block(effect);
        effect->applyParameters(want.parameters);
    }
}

} // namespace SettingsImporter
