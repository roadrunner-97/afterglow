#ifndef GRIDVIEW_H
#define GRIDVIEW_H

#include <QHash>
#include <QImage>
#include <QString>
#include <QWidget>

class QListWidget;
class QListWidgetItem;
class QTimer;

class GridView : public QWidget {
    Q_OBJECT
public:
    // Stored as the persisted single-char code in .afterglow-catalog.json.
    // Pressing the same key twice cycles back to None (the resting state).
    enum class Mark : char { None = 0, Accept = 'A', Refine = 'R', Decline = 'D' };

    enum class ProofStatus { NotProofed, Proofing, Proofed };

    explicit GridView(QWidget *parent = nullptr);

    // Replace the displayed list. Items show a placeholder icon until
    // setThumbnail() is called for each path.
    void setPhotos(const QStringList &paths);

    // Update the thumbnail for a single photo. Caller decodes off-thread
    // and pushes results in via this method (safe on the GUI thread).
    void   setThumbnail(const QString &path, const QImage &thumb);
    QImage thumbnail(const QString &path) const;

    // Keep the visible Gallery selection synchronized with Loupe navigation.
    bool    setCurrentPath(const QString &path);
    QString currentPath() const;

    // Set/get a photo's triage mark. Caller persists this — GridView only
    // displays it (colored background).
    void setMark(const QString &path, Mark m);
    Mark mark(const QString &path) const;
    void toggleMark(const QString &path, Mark requested);

    // Update the proof-status dot for a single photo.
    void setProofStatus(const QString &path, ProofStatus status);

    // Show whether a photo has committed Develop adjustments. The delegate
    // renders a persistent pencil badge independently of transient proofing.
    void setEdited(const QString &path, bool edited);
    bool isEdited(const QString &path) const;

signals:
    void photoActivated(const QString &path);
    void markChanged(const QString &path, Mark m);
    void currentPathChanged(const QString &path);
    void copySettingsRequested(const QString &path);
    void pasteSettingsRequested(const QString &path);

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private:
    void applyIconSize();
    void resizeThumbnails(QWheelEvent *event);
    void ensureAnimTimer();
    void onAnimTick();

    QListWidget *m_list       = nullptr;
    int          m_iconPx     = 160;
    float        m_pulsePhase = 0.0f;
    QTimer      *m_animTimer  = nullptr;

    QHash<QString, Mark>        m_marks;
    QHash<QString, ProofStatus> m_proofStatus;
    QHash<QString, bool>        m_edited;
    QHash<QString, float>       m_fadeOpacity;
};

#endif // GRIDVIEW_H
