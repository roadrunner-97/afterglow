#ifndef EFFECTORGANIZERWIDGET_H
#define EFFECTORGANIZERWIDGET_H

#include <QWidget>

class EffectManager;
class QListWidget;

class EffectOrganizerWidget : public QWidget {
    Q_OBJECT

public:
    explicit EffectOrganizerWidget(EffectManager *effects, QWidget *parent = nullptr);

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
