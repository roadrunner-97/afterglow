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

    int                 addFrames(const QStringList &paths);
    QVector<StackFrame> frames() const;
    QString             referencePath() const;
    void                setReferencePath(const QString &path);
    QString             currentFramePath() const;
    void                setFramePreview(const QString &path, const QImage &preview);
    void                setResult(const QImage &result);
    QImage              result() const;
    void                setBuilding(bool building, int totalFrames = 0);
    void                setProgress(int completedFrames, int totalFrames, const QString &path);
    void                setStatus(const QString &message);

signals:
    void addFramesRequested();
    void addCurrentFolderRawsRequested();
    void currentFrameChanged(QString path);
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
    void showSelectedFrame();
    void showStackResult();

    enum class PreviewMode { Frame, Result };

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
    QPushButton  *m_showFrame      = nullptr;
    QPushButton  *m_showResult     = nullptr;
    QString       m_referencePath;
    QString       m_currentFramePath;
    QImage        m_framePreview;
    QImage        m_result;
    PreviewMode   m_previewMode = PreviewMode::Frame;
};

#endif // STACKWORKSPACE_H
