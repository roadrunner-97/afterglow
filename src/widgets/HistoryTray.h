#ifndef HISTORYTRAY_H
#define HISTORYTRAY_H

#include <QString>
#include <QVector>
#include <QWidget>

class QListWidget;
class QListWidgetItem;
class QToolButton;

class HistoryTray : public QWidget {
    Q_OBJECT
public:
    struct Row { QString label; };

    explicit HistoryTray(QWidget* parent = nullptr);

    // Full rebuild. rows correspond to UndoHistory::entries(); a synthetic
    // "Original" row is prepended at list index 0. cursor is the history
    // cursor: list item at index `cursor` is the current state.
    void setHistory(const QVector<Row>& rows, int cursor);

signals:
    void rowActivated(int index);   // user clicked row; index 0 = "Original"

private:
    void toggleCollapsed();

    QListWidget* m_list        = nullptr;
    QToolButton* m_collapseBtn = nullptr;
    bool         m_collapsed   = false;
};

#endif // HISTORYTRAY_H
