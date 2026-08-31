#include "EffectOrganizerDialog.h"
#include "EffectManager.h"
#include <QAbstractItemView>
#include <QLabel>
#include <QListWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QHBoxLayout>

namespace {
QListWidget *makeEffectList(const char *objectName, QWidget *parent) {
    auto *list = new QListWidget(parent);
    list->setObjectName(objectName);
    list->setDragEnabled(true);
    list->setAcceptDrops(true);
    list->setDropIndicatorShown(true);
    list->setDragDropMode(QAbstractItemView::DragDrop);
    list->setDefaultDropAction(Qt::MoveAction);
    list->setSelectionMode(QAbstractItemView::SingleSelection);
    list->setMinimumWidth(220);
    return list;
}
} // namespace

EffectOrganizerWidget::EffectOrganizerWidget(EffectManager *effects, QWidget *parent)
    : QWidget(parent), m_effects(effects), m_available(makeEffectList("availableEffectsList", this)),
      m_enabled(makeEffectList("enabledEffectsList", this)) {
    setObjectName("effectOrganizerPage");

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(12, 6, 6, 6);
    layout->setSpacing(10);
    auto *hint = new QLabel("Drag effects between the lists. Drag within Enabled Effects to set processing order.");
    hint->setWordWrap(true);
    layout->addWidget(hint);

    auto *columns   = new QHBoxLayout();
    auto  addColumn = [columns](const QString &title, QListWidget *list) {
        auto *column = new QVBoxLayout();
        auto *label  = new QLabel(title);
        QFont font   = label->font();
        font.setBold(true);
        label->setFont(font);
        column->addWidget(label);
        column->addWidget(list, 1);
        columns->addLayout(column, 1);
    };
    addColumn("Available Effects", m_available);
    addColumn("Enabled Effects", m_enabled);
    layout->addLayout(columns, 1);

    auto scheduleApply = [this]() {
        if (m_rebuilding || m_applyPending) return;
        m_applyPending = true;
        QTimer::singleShot(0, this, [this]() {
            m_applyPending = false;
            applyLists();
        });
    };
    connect(m_available->model(), &QAbstractItemModel::rowsInserted, this, scheduleApply);
    connect(m_available->model(), &QAbstractItemModel::rowsMoved, this, scheduleApply);
    connect(m_available->model(), &QAbstractItemModel::rowsRemoved, this, scheduleApply);
    connect(m_enabled->model(), &QAbstractItemModel::rowsInserted, this, scheduleApply);
    connect(m_enabled->model(), &QAbstractItemModel::rowsMoved, this, scheduleApply);
    connect(m_enabled->model(), &QAbstractItemModel::rowsRemoved, this, scheduleApply);
    connect(m_effects, &EffectManager::effectToggled, this, [this]() { rebuild(); });
    connect(m_effects, &EffectManager::effectsReordered, this, &EffectOrganizerWidget::rebuild);
    rebuild();
}

void EffectOrganizerWidget::rebuild() {
    if (m_rebuilding) return;
    m_rebuilding = true;
    m_available->clear();
    m_enabled->clear();
    for (const EffectEntry &entry : m_effects->entries()) {
        if (!entry.effect) continue;
        auto *item = new QListWidgetItem(entry.effect->getName());
        item->setData(Qt::UserRole, entry.effect->getId());
        (entry.enabled ? m_enabled : m_available)->addItem(item);
    }
    m_rebuilding = false;
}

void EffectOrganizerWidget::applyLists() {
    if (m_rebuilding) return;
    QVector<QPair<QString, bool>> configuration;
    configuration.reserve(m_enabled->count() + m_available->count());
    for (int i = 0; i < m_enabled->count(); ++i)
        configuration.append({m_enabled->item(i)->data(Qt::UserRole).toString(), true});
    for (int i = 0; i < m_available->count(); ++i)
        configuration.append({m_available->item(i)->data(Qt::UserRole).toString(), false});
    m_effects->configureEffects(configuration);
    emit organizationChanged();
}
