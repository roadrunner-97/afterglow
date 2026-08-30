#ifndef UISERVICES_H
#define UISERVICES_H

#include "ExportOptions.h"

#include <QMessageBox>
#include <QString>
#include <memory>
#include <optional>

class QWidget;

// Boundary around modal/native UI. Tests inject a deterministic implementation
// while production uses Qt's dialogs and message boxes.
class UiServices {
public:
    virtual ~UiServices() = default;

    virtual QString openFile(QWidget *parent, const QString &title, const QString &initialPath,
                             const QString &filter)                                                    = 0;
    virtual QString saveFile(QWidget *parent, const QString &title, const QString &initialPath,
                             const QString &filter)                                                    = 0;
    virtual QString chooseDirectory(QWidget *parent, const QString &title, const QString &initialPath) = 0;
    virtual std::optional<ExportOptions::Options> chooseExportOptions(QWidget       *parent,
                                                                      const QString &defaultDirectory) = 0;

    virtual void information(QWidget *parent, const QString &title, const QString &message) = 0;
    virtual void warning(QWidget *parent, const QString &title, const QString &message)     = 0;
    virtual bool confirm(QWidget *parent, const QString &title, const QString &message)     = 0;
};

std::unique_ptr<UiServices> createNativeUiServices();

#endif // UISERVICES_H
