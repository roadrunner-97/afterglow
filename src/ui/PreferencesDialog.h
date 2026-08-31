#ifndef PREFERENCESDIALOG_H
#define PREFERENCESDIALOG_H

#include <QDialog>

class EffectManager;
class QComboBox;
class QListWidget;
class QStackedWidget;

class PreferencesDialog : public QDialog {
    Q_OBJECT

public:
    explicit PreferencesDialog(EffectManager *effects, QWidget *parent = nullptr);
    void showPage(int index);

signals:
    void effectOrganizationChanged();
    void gpuDeviceChanged();

private:
    QListWidget    *m_pages;
    QStackedWidget *m_stack;
    QComboBox      *m_gpuSelector;
};

#endif
