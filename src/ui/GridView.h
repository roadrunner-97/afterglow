#ifndef GRIDVIEW_H
#define GRIDVIEW_H

#include <QWidget>
#include <QString>
#include <QImage>
#include <QHash>

class QListWidget;
class QListWidgetItem;

class GridView : public QWidget {
    Q_OBJECT
public:
    enum class Mark : char { None = 'U', Pick = 'P', Reject = 'X' };

    explicit GridView(QWidget* parent = nullptr);

    // Replace the displayed list. Items show a placeholder icon until
    // setThumbnail() is called for each path.
    void setPhotos(const QStringList& paths);

    // Update the thumbnail for a single photo. Caller decodes off-thread
    // and pushes results in via this method (safe on the GUI thread).
    void setThumbnail(const QString& path, const QImage& thumb);

    // Set/get a photo's triage mark. Caller persists this — GridView only
    // displays it (colored border).
    void setMark(const QString& path, Mark m);
    Mark mark(const QString& path) const;

signals:
    // Emitted on double-click or Enter on a cell.
    void photoActivated(const QString& path);

    // Emitted when the user presses P/X/U with a cell selected.
    void markChanged(const QString& path, Mark m);

protected:
    void wheelEvent(QWheelEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:
    QListWidget* m_list;
    QHash<QString, Mark> m_marks;
    int m_iconPx = 160;  // initial cell edge in pixels
    void applyIconSize();
};

#endif // GRIDVIEW_H
