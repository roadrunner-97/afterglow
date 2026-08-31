#ifndef UNDOHISTORY_H
#define UNDOHISTORY_H

#include "SettingsImporter.h"
#include <QHash>
#include <QMap>
#include <QObject>
#include <QVariant>
#include <optional>
#include <utility>

class UndoHistory : public QObject {
    Q_OBJECT
public:
    struct ParamDelta {
        QVariant from;
        QVariant to;
    };

    struct Entry {
        QString                              effectId;
        std::optional<std::pair<bool, bool>> enabled; // first=from, second=to
        QMap<QString, ParamDelta>            params;
        bool                                 empty() const {
            return !enabled.has_value() && params.isEmpty();
        }
    };

    explicit UndoHistory(int capacity = 200, QObject *parent = nullptr);

    // Replace shadow and clear log. Call after loading a new image with no history sidecar.
    void seed(const QVector<SettingsImporter::EffectSettings> &current);

    // Diff current state against shadow; push per-effect entries and advance
    // cursor if anything changed. Truncates any redo tail. No-op while isApplying().
    void recordFromCurrent(const QVector<SettingsImporter::EffectSettings> &current);
    void ensureTracked(const SettingsImporter::EffectSettings &current);

    bool canUndo() const;
    bool canRedo() const;

    // Returns the entry to apply (caller applies from/to values); moves cursor and updates shadow.
    std::optional<Entry> undo();
    std::optional<Entry> redo();

    // Guard so applying an undo/redo doesn't get recorded as a new commit.
    void setApplying(bool b);
    bool isApplying() const;

    // Serializer access.
    const QVector<Entry> &entries() const;
    int                   cursor() const;
    void                  load(QVector<Entry> entries, int cursor, QVector<SettingsImporter::EffectSettings> shadow);

signals:
    void canUndoChanged(bool);
    void canRedoChanged(bool);
    void historyChanged();

private:
    using Shadow = QHash<QString, SettingsImporter::EffectSettings>;

    static Shadow buildShadow(const QVector<SettingsImporter::EffectSettings> &v);
    void          updateShadowFrom(const Entry &e);
    void          updateShadowTo(const Entry &e);

    int            m_capacity;
    QVector<Entry> m_entries;
    int            m_cursor = 0;
    Shadow         m_shadow;
    bool           m_applying = false;
};

#endif // UNDOHISTORY_H
