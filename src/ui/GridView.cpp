#include "GridView.h"

#include <QListWidget>
#include <QListWidgetItem>
#include <QPainter>
#include <QStyledItemDelegate>
#include <QStyleOptionViewItem>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QFileInfo>
#include <QPixmap>

namespace {

// Paints the base list item then overlays a small status circle in the
// bottom-right corner of each thumbnail to show proof state.
class ProofStatusDelegate : public QStyledItemDelegate {
public:
    explicit ProofStatusDelegate(const QHash<QString, GridView::ProofStatus>* status,
                                  QObject* parent = nullptr)
        : QStyledItemDelegate(parent), m_status(status) {}

    void paint(QPainter* painter, const QStyleOptionViewItem& option,
               const QModelIndex& index) const override {
        QStyledItemDelegate::paint(painter, option, index);

        const QString path = index.data(Qt::UserRole).toString();
        const GridView::ProofStatus status =
            m_status->value(path, GridView::ProofStatus::NotProofed);

        QColor color;
        switch (status) {
            case GridView::ProofStatus::NotProofed: color = QColor(130, 130, 130, 200); break;
            case GridView::ProofStatus::Proofing:   color = QColor(255, 195, 40,  220); break;
            case GridView::ProofStatus::Proofed:    color = QColor(72,  200, 100, 220); break;
        }

        constexpr int R = 5;
        const QPoint centre(option.rect.right() - R - 5,
                            option.rect.bottom() - R - 5);
        painter->save();
        painter->setRenderHint(QPainter::Antialiasing);
        painter->setPen(Qt::NoPen);
        painter->setBrush(color);
        painter->drawEllipse(centre, R, R);
        painter->restore();
    }

private:
    const QHash<QString, GridView::ProofStatus>* m_status;
};

} // namespace

GridView::GridView(QWidget* parent)
    : QWidget(parent)
{
    m_list = new QListWidget(this);
    m_list->setItemDelegate(new ProofStatusDelegate(&m_proofStatus, this));
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

    // Track selection changes so the host can sync m_currentImagePath as
    // the user clicks or arrow-keys through the grid.
    connect(m_list, &QListWidget::currentItemChanged, this,
            [this](QListWidgetItem* current, QListWidgetItem*) {
                emit currentPathChanged(current ? current->data(Qt::UserRole).toString()
                                                : QString());
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
    m_proofStatus.clear();

    for (const QString& path : paths) {
        QListWidgetItem* item = new QListWidgetItem(m_list);
        item->setText(QFileInfo(path).fileName());
        item->setData(Qt::UserRole, path);
        // Set a placeholder icon (empty icon until setThumbnail is called)
        item->setIcon(QIcon());
    }

    // Default the cursor to the first item so the toolbar's Develop /
    // Loupe buttons have something to act on without the user having to
    // click first.
    if (m_list->count() > 0)
        m_list->setCurrentRow(0);
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
                case Mark::Accept:
                    item->setBackground(QColor(144, 238, 144, 100));  // light green
                    break;
                case Mark::Refine:
                    item->setBackground(QColor(240, 210, 120, 110));  // warm amber
                    break;
                case Mark::Decline:
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

void GridView::setProofStatus(const QString& path, ProofStatus status)
{
    m_proofStatus[path] = status;
    for (int i = 0; i < m_list->count(); ++i) {
        QListWidgetItem* item = m_list->item(i);
        if (item->data(Qt::UserRole).toString() == path) {
            m_list->update(m_list->indexFromItem(item));
            break;
        }
    }
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

    auto applyMark = [&](Mark requested) {
        // Pressing the same letter as the current mark toggles it off
        // (back to None) — matches the "exclusive but defaults to none"
        // behaviour the Loupe sidebar buttons present.
        const Mark next = (mark(currentPath) == requested) ? Mark::None : requested;
        setMark(currentPath, next);
        emit markChanged(currentPath, next);
        m_list->setCurrentRow(m_list->currentRow() + 1);
        event->accept();
    };

    if (event->key() == Qt::Key_A) {
        applyMark(Mark::Accept);
    } else if (event->key() == Qt::Key_R) {
        applyMark(Mark::Refine);
    } else if (event->key() == Qt::Key_D) {
        applyMark(Mark::Decline);
    } else if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
        emit photoActivated(currentPath);
        event->accept();
    } else {
        QWidget::keyPressEvent(event);
    }
}
