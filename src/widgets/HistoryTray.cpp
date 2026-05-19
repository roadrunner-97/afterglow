#include "HistoryTray.h"
#include <QFont>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QPalette>
#include <QToolButton>
#include <QVBoxLayout>

HistoryTray::HistoryTray(QWidget* parent)
    : QWidget(parent)
{
    setAutoFillBackground(true);

    auto* outerLayout = new QVBoxLayout(this);
    outerLayout->setContentsMargins(0, 0, 0, 0);
    outerLayout->setSpacing(0);

    // Header bar
    auto* header = new QWidget();
    auto* headerLayout = new QHBoxLayout(header);
    headerLayout->setContentsMargins(4, 2, 4, 2);
    auto* titleLabel = new QLabel("History");
    QFont titleFont = titleLabel->font();
    titleFont.setBold(true);
    titleLabel->setFont(titleFont);
    headerLayout->addWidget(titleLabel, 1);
    m_collapseBtn = new QToolButton();
    m_collapseBtn->setText(u8"▲");   // ▲
    m_collapseBtn->setAutoRaise(true);
    headerLayout->addWidget(m_collapseBtn);
    outerLayout->addWidget(header);

    // History list
    m_list = new QListWidget();
    m_list->setFocusPolicy(Qt::NoFocus);
    m_list->setSelectionMode(QAbstractItemView::SingleSelection);
    m_list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_list->setMaximumHeight(200);
    outerLayout->addWidget(m_list);

    setFixedWidth(fontMetrics().averageCharWidth() * 22);

    connect(m_collapseBtn, &QToolButton::clicked, this, &HistoryTray::toggleCollapsed);
    connect(m_list, &QListWidget::itemClicked, this, [this](QListWidgetItem* item) {
        emit rowActivated(m_list->row(item));
    });
}

void HistoryTray::setHistory(const QVector<Row>& rows, int cursor)
{
    m_list->clear();

    m_list->addItem("Original");
    for (const auto& r : rows)
        m_list->addItem(r.label);

    const QColor dimColor = palette().color(QPalette::Disabled, QPalette::Text);
    const int total = m_list->count();
    for (int i = 0; i < total; ++i) {
        QListWidgetItem* item = m_list->item(i);
        if (i == cursor) {
            QFont f = item->font();
            f.setBold(true);
            item->setFont(f);
        } else if (i > cursor) {
            item->setForeground(dimColor);
        }
    }

    if (cursor >= 0 && cursor < total) {
        m_list->setCurrentRow(cursor);
        m_list->scrollToItem(m_list->item(cursor));
    }
}

void HistoryTray::toggleCollapsed()
{
    m_collapsed = !m_collapsed;
    m_list->setVisible(!m_collapsed);
    m_collapseBtn->setText(m_collapsed ? u8"▼" : u8"▲");  // ▼ / ▲
    adjustSize();
}
