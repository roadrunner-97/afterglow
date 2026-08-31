#ifndef EFFECTORGANIZERDIALOG_H
#define EFFECTORGANIZERDIALOG_H

#include <QDialog>

class EffectManager;
class QListWidget;

class EffectOrganizerDialog : public QDialog {
    Q_OBJECT

public:
    explicit EffectOrganizerDialog(EffectManager *effects, QWidget *parent = nullptr);

signals:
    void organizationChanged();

private:
    void rebuild();
    void applyLists();

    EffectManager *m_effects;
    QListWidget   *m_available;
    QListWidget   *m_enabled;
    bool           m_rebuilding   = false;
    bool           m_applyPending = false;
};

#endif
