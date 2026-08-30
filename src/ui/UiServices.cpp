#include "UiServices.h"

#include "ExportDialog.h"

#include <QFileDialog>

namespace {

class NativeUiServices final : public UiServices {
public:
    QString openFile(QWidget *parent, const QString &title, const QString &initialPath,
                     const QString &filter) override {
        return QFileDialog::getOpenFileName(parent, title, initialPath, filter);
    }

    QString saveFile(QWidget *parent, const QString &title, const QString &initialPath,
                     const QString &filter) override {
        return QFileDialog::getSaveFileName(parent, title, initialPath, filter);
    }

    QString chooseDirectory(QWidget *parent, const QString &title, const QString &initialPath) override {
        return QFileDialog::getExistingDirectory(parent, title, initialPath, QFileDialog::ShowDirsOnly);
    }

    std::optional<ExportOptions::Options> chooseExportOptions(QWidget       *parent,
                                                              const QString &defaultDirectory) override {
        ExportDialog dialog(parent);
        dialog.setDefaultDestinationDir(defaultDirectory);
        if (dialog.exec() != QDialog::Accepted) return std::nullopt;
        return dialog.options();
    }

    void information(QWidget *parent, const QString &title, const QString &message) override {
        QMessageBox::information(parent, title, message);
    }

    void warning(QWidget *parent, const QString &title, const QString &message) override {
        QMessageBox::warning(parent, title, message);
    }

    bool confirm(QWidget *parent, const QString &title, const QString &message) override {
        return QMessageBox::question(parent, title, message) == QMessageBox::Yes;
    }
};

} // namespace

std::unique_ptr<UiServices> createNativeUiServices() {
    return std::make_unique<NativeUiServices>();
}
