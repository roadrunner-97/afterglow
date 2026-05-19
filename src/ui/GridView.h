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
    // Stored as the persisted single-char code in .afterglow-catalog.json.
    // Pressing the same key twice cycles back to None (the resting state).
    enum class Mark : char { None = 0, Accept = 'A', Refine = 'R', Decline = 'D' };

    // Proof status shown as a small dot in each thumbnail's corner.
    enum class ProofStatus { NotProofed, Proofing, Proofed };

    explicit GridView(QWidget *parent = nullptr);

    // Replace the displayed list. Items show a placeholder icon until
    // setThumbnail() is called for each path.
    void setPhotos(const QStringList &paths);

    // Update the thumbnail for a single photo. Caller decodes off-thread
    // and pushes results in via this method (safe on the GUI thread).
    void setThumbnail(const QString &path, const QImage &thumb);

    // Set/get a photo's triage mark. Caller persists this — GridView only
    // displays it (colored border).
    void setMark(const QString &path, Mark m);
    Mark mark(const QString &path) const;

    // Update the proof status dot for a single photo.
    void setProofStatus(const QString &path, ProofStatus status);

signals:
    // Emitted on double-click or Enter on a cell.
    void photoActivated(const QString &path);

    // Emitted when the user presses P/X/U with a cell selected.
    void markChanged(const QString &path, Mark m);

    // Emitted as the user moves the selection cursor (single click or
    // arrow keys).  Empty path when nothing is selected.
    void currentPathChanged(const QString &path);

protected:
    void wheelEvent(QWheelEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private:
    QListWidget                *m_list;
    QHash<QString, Mark>        m_marks;
    QHash<QString, ProofStatus> m_proofStatus;
    int                         m_iconPx = 160; // initial cell edge in pixels
    void                        applyIconSize();
};

#endif // GRIDVIEW_H
