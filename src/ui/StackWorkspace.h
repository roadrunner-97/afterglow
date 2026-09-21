#ifndef STACKWORKSPACE_H
#define STACKWORKSPACE_H

#include "LongExposureStack.h"

#include <QImage>
#include <QWidget>

class QLabel;
class QListWidget;
class QListWidgetItem;
class QProgressBar;
class QPushButton;
class QResizeEvent;

class StackWorkspace : public QWidget {
    Q_OBJECT

public:
    explicit StackWorkspace(QWidget *parent = nullptr);

    void                addFrames(const QStringList &paths);
    QVector<StackFrame> frames() const;
    QString             referencePath() const;
    void                setReferencePath(const QString &path);
    void                setResult(const QImage &result);
    QImage              result() const;
    void                setBuilding(bool building, int totalFrames = 0);
    void                setProgress(int completedFrames, int totalFrames, const QString &path);
    void                setStatus(const QString &message);

signals:
    void addFramesRequested();
    void editReferenceRequested(QString path);
    void rebuildRequested();
    void cancelRequested();
    void saveRequested();

protected:
    void resizeEvent(QResizeEvent *event) override;

private:
    void updateFrameSummary();
    void updateReferencePresentation();
    void updatePreviewPixmap();

    QListWidget  *m_frames         = nullptr;
    QLabel       *m_frameSummary   = nullptr;
    QLabel       *m_referenceLabel = nullptr;
    QLabel       *m_preview        = nullptr;
    QLabel       *m_status         = nullptr;
    QProgressBar *m_progress       = nullptr;
    QPushButton  *m_setReference   = nullptr;
    QPushButton  *m_editReference  = nullptr;
    QPushButton  *m_rebuild        = nullptr;
    QPushButton  *m_cancel         = nullptr;
    QPushButton  *m_save           = nullptr;
    QString       m_referencePath;
    QImage        m_result;
};

#endif // STACKWORKSPACE_H
