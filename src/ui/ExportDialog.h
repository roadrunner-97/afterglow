#ifndef EXPORTDIALOG_H
#define EXPORTDIALOG_H

#include <QDialog>
#include "ExportOptions.h"

class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSlider;

// Modal dialog that collects an ExportOptions::Options for batch-friendly
// image export.  Tied to no specific source file: filename templating and the
// destination folder are independent inputs.  Chosen options are persisted to
// QSettings ("export/...") so subsequent invocations preload the last choice.
class ExportDialog : public QDialog {
    Q_OBJECT

public:
    explicit ExportDialog(QWidget* parent = nullptr);

    // Pre-populate the destination folder when the user has no saved choice
    // yet (e.g. first launch).  Pattern, format, quality, and conflict policy
    // come from QSettings or hard defaults.
    void setDefaultDestinationDir(const QString& dir);

    ExportOptions::Options options() const;

private slots:
    void browseForDirectory();
    void onFormatChanged(int idx);

private:
    void loadFromSettings();
    void persistToSettings() const;
    void accept() override;

    QLineEdit*   m_destEdit       = nullptr;
    QPushButton* m_browseBtn      = nullptr;
    QLineEdit*   m_patternEdit    = nullptr;
    QComboBox*   m_formatCombo    = nullptr;
    QSlider*     m_qualitySlider  = nullptr;
    QLabel*      m_qualityLabel   = nullptr;
    QComboBox*   m_conflictCombo  = nullptr;
};

#endif // EXPORTDIALOG_H
