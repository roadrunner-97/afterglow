#include "GridView.h"

#include <QListWidget>
#include <QListWidgetItem>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QFileInfo>
#include <QPixmap>

GridView::GridView(QWidget* parent)
    : QWidget(parent)
{
    m_list = new QListWidget(this);
    m_list->setViewMode(QListView::IconMode);
    m_list->setResizeMode(QListView::Adjust);
    m_list->setMovement(QListView::Static);
    m_list->setWrapping(true);
    m_list->setFocusPolicy(Qt::StrongFocus);
    m_list->setFocusProxy(m_list);

    // Connect item double-click to photoActivated signal
    connect(m_list, &QListWidget::itemDoubleClicked, this,
            [this](QListWidgetItem* item) {
                QString path = item->data(Qt::UserRole).toString();
                emit photoActivated(path);
            });

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_list);
    setLayout(layout);

    setFocusProxy(m_list);
    applyIconSize();
}

void GridView::setPhotos(const QStringList& paths)
{
    m_list->clear();
    m_marks.clear();

    for (const QString& path : paths) {
        QListWidgetItem* item = new QListWidgetItem(m_list);
        item->setText(QFileInfo(path).fileName());
        item->setData(Qt::UserRole, path);
        // Set a placeholder icon (empty icon until setThumbnail is called)
        item->setIcon(QIcon());
    }
}

void GridView::setThumbnail(const QString& path, const QImage& thumb)
{
    for (int i = 0; i < m_list->count(); ++i) {
        QListWidgetItem* item = m_list->item(i);
        if (item->data(Qt::UserRole).toString() == path) {
            QPixmap pixmap = QPixmap::fromImage(thumb);
            item->setIcon(QIcon(pixmap));
            break;
        }
    }
}

void GridView::setMark(const QString& path, Mark m)
{
    m_marks[path] = m;

    // Find and update the item's background color based on the mark
    for (int i = 0; i < m_list->count(); ++i) {
        QListWidgetItem* item = m_list->item(i);
        if (item->data(Qt::UserRole).toString() == path) {
            switch (m) {
                case Mark::Pick:
                    item->setBackground(QColor(144, 238, 144, 100));  // light green
                    break;
                case Mark::Reject:
                    item->setBackground(QColor(255, 127, 127, 100));  // light red
                    break;
                case Mark::None:
                    item->setBackground(QColor(255, 255, 255, 0));  // transparent
                    break;
            }
            break;
        }
    }
}

GridView::Mark GridView::mark(const QString& path) const
{
    return m_marks.value(path, Mark::None);
}

void GridView::applyIconSize()
{
    m_list->setIconSize(QSize(m_iconPx, m_iconPx));
    m_list->setGridSize(QSize(m_iconPx + 16, m_iconPx + 32));
}

void GridView::wheelEvent(QWheelEvent* event)
{
    if (event->modifiers() & Qt::ControlModifier) {
        int delta = event->angleDelta().y() / 8;
        m_iconPx += delta;
        m_iconPx = qBound(48, m_iconPx, 512);
        applyIconSize();
        event->accept();
    } else {
        QWidget::wheelEvent(event);
    }
}

void GridView::keyPressEvent(QKeyEvent* event)
{
    QListWidgetItem* currentItem = m_list->currentItem();
    if (!currentItem) {
        QWidget::keyPressEvent(event);
        return;
    }

    QString currentPath = currentItem->data(Qt::UserRole).toString();

    if (event->key() == Qt::Key_P) {
        setMark(currentPath, Mark::Pick);
        emit markChanged(currentPath, Mark::Pick);
        m_list->setCurrentRow(m_list->currentRow() + 1);
        event->accept();
    } else if (event->key() == Qt::Key_X) {
        setMark(currentPath, Mark::Reject);
        emit markChanged(currentPath, Mark::Reject);
        m_list->setCurrentRow(m_list->currentRow() + 1);
        event->accept();
    } else if (event->key() == Qt::Key_U) {
        setMark(currentPath, Mark::None);
        emit markChanged(currentPath, Mark::None);
        m_list->setCurrentRow(m_list->currentRow() + 1);
        event->accept();
    } else if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
        emit photoActivated(currentPath);
        event->accept();
    } else {
        QWidget::keyPressEvent(event);
    }
}
