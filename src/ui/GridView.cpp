#include "GridView.h"

#include <QListWidget>
#include <QListWidgetItem>
#include <QPainter>
#include <QStyledItemDelegate>
#include <QStyleOptionViewItem>
#include <QTimer>
#include <QMenu>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QFileInfo>
#include <QPixmap>
#include <QEvent>

#include <algorithm>
#include <cmath>

namespace {

constexpr float kPi = 3.14159265358979323846f;

constexpr int kCardPadding = 8;

// Paints thumbnails into a predictable card rectangle. Proof state is a small
// status dot; edited photos get a compact pencil glyph.
class GalleryItemDelegate : public QStyledItemDelegate {
public:
    explicit GalleryItemDelegate(const QHash<QString, GridView::ProofStatus> *status,
                                 const QHash<QString, bool> *edited, const QHash<QString, float> *fadeOpacity,
                                 const float *pulsePhase, QObject *parent = nullptr)
        : QStyledItemDelegate(parent), m_status(status), m_edited(edited), m_fadeOpacity(fadeOpacity),
          m_pulsePhase(pulsePhase) {}

    void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override {
        const QString               path   = index.data(Qt::UserRole).toString();
        const GridView::ProofStatus status = m_status->value(path, GridView::ProofStatus::NotProofed);

        painter->save();
        painter->setRenderHint(QPainter::Antialiasing);

        if (option.state & QStyle::State_Selected) {
            painter->setPen(Qt::NoPen);
            painter->setBrush(option.palette.highlight().color().lighter(125));
            painter->drawRoundedRect(option.rect.adjusted(2, 2, -2, -2), 5, 5);
        } else if (index.data(Qt::BackgroundRole).canConvert<QBrush>()) {
            painter->setPen(Qt::NoPen);
            painter->setBrush(qvariant_cast<QBrush>(index.data(Qt::BackgroundRole)));
            painter->drawRoundedRect(option.rect.adjusted(2, 2, -2, -2), 5, 5);
        }

        const QRect   available = option.rect.adjusted(kCardPadding, kCardPadding, -kCardPadding, -kCardPadding);
        const QIcon   icon      = qvariant_cast<QIcon>(index.data(Qt::DecorationRole));
        const QPixmap source    = icon.pixmap(option.decorationSize);
        QRect         imageRect = available;
        if (!source.isNull()) {
            const QSize fitted = source.size().scaled(available.size(), Qt::KeepAspectRatio);
            imageRect          = QRect(QPoint(0, 0), fitted);
            imageRect.moveCenter(available.center());
            painter->drawPixmap(imageRect, source);
        }

        QColor baseColor;
        float  opacity = 1.0f;
        bool   drawDot = true;

        switch (status) {
        case GridView::ProofStatus::NotProofed:
            baseColor = QColor(210, 50, 50);
            break;
        case GridView::ProofStatus::Proofing:
            baseColor = QColor(255, 155, 25);
            // Pulse opacity 0.45 → 1.0 → 0.45 on a sine wave
            opacity = 0.45f + 0.55f * (0.5f + 0.5f * std::sin(*m_pulsePhase * 2.0f * kPi));
            break;
        case GridView::ProofStatus::Proofed: {
            float fade = m_fadeOpacity->value(path, 0.0f);
            if (fade <= 0.0f) {
                drawDot = false;
                break;
            }
            baseColor = QColor(255, 155, 25);
            opacity   = fade;
            break;
        }
        }

        if (drawDot) {
            baseColor.setAlphaF(opacity);
            const QPoint centre(imageRect.right() - 7, imageRect.top() + 8);
            painter->setPen(QPen(QColor(20, 20, 20, 150), 2));
            painter->setBrush(baseColor);
            painter->drawEllipse(centre, 5, 5);
        }

        if (m_edited->value(path, false)) {
            painter->setPen(Qt::NoPen);
            painter->setBrush(QColor(30, 30, 30, 210));
            const QRect badge(imageRect.right() - 25, imageRect.bottom() - 25, 20, 20);
            painter->drawEllipse(badge);
            painter->setPen(QPen(Qt::white, 2.2, Qt::SolidLine, Qt::RoundCap));
            painter->drawLine(badge.left() + 6, badge.bottom() - 6, badge.right() - 5, badge.top() + 5);
            painter->setPen(QPen(Qt::white, 1.4, Qt::SolidLine, Qt::RoundCap));
            painter->drawLine(badge.left() + 5, badge.bottom() - 5, badge.left() + 8, badge.bottom() - 6);
        }

        painter->restore();
    }

    QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &) const override {
        return option.decorationSize + QSize(kCardPadding * 2, kCardPadding * 2);
    }

private:
    const QHash<QString, GridView::ProofStatus> *m_status;
    const QHash<QString, bool>                  *m_edited;
    const QHash<QString, float>                 *m_fadeOpacity;
    const float                                 *m_pulsePhase;
};

} // namespace

GridView::GridView(QWidget *parent) : QWidget(parent) {
    m_list = new QListWidget(this);
    m_list->setItemDelegate(new GalleryItemDelegate(&m_proofStatus, &m_edited, &m_fadeOpacity, &m_pulsePhase, this));
    m_list->setViewMode(QListView::IconMode);
    m_list->setResizeMode(QListView::Adjust);
    m_list->setMovement(QListView::Static);
    m_list->setWrapping(true);
    m_list->setFocusPolicy(Qt::StrongFocus);
    m_list->setFocusProxy(m_list);
    m_list->setSpacing(4);
    m_list->setUniformItemSizes(true);
    m_list->viewport()->installEventFilter(this);
    m_list->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_list, &QListWidget::customContextMenuRequested, this, [this](const QPoint &pos) {
        QListWidgetItem *item = m_list->itemAt(pos);
        if (!item) return;
        m_list->setCurrentItem(item);
        const QString path = item->data(Qt::UserRole).toString();
        QMenu         menu(m_list);
        QAction      *copy  = menu.addAction("Copy Develop Settings");
        QAction      *paste = menu.addAction("Paste Develop Settings");
        menu.addSeparator();
        QMenu *markMenu      = menu.addMenu("Mark");
        auto   addMarkAction = [this, markMenu, &path](const QString &label, Mark requested) {
            QAction *action = markMenu->addAction(label);
            action->setCheckable(true);
            action->setChecked(mark(path) == requested);
            connect(action, &QAction::triggered, this, [this, path, requested]() { toggleMark(path, requested); });
        };
        addMarkAction("Accept", Mark::Accept);
        addMarkAction("Refine", Mark::Refine);
        addMarkAction("Decline", Mark::Decline);
        QAction *chosen = menu.exec(m_list->viewport()->mapToGlobal(pos));
        if (chosen == copy) emit copySettingsRequested(path);
        else if (chosen == paste) emit pasteSettingsRequested(path);
    });

    connect(m_list, &QListWidget::itemDoubleClicked, this,
            [this](QListWidgetItem *item) { emit photoActivated(item->data(Qt::UserRole).toString()); });

    connect(m_list, &QListWidget::currentItemChanged, this, [this](QListWidgetItem *current, QListWidgetItem *) {
        emit currentPathChanged(current ? current->data(Qt::UserRole).toString() : QString());
    });

    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_list);
    setLayout(layout);

    setFocusProxy(m_list);
    applyIconSize();
}

void GridView::setPhotos(const QStringList &paths) {
    m_list->clear();
    m_marks.clear();
    m_proofStatus.clear();
    m_edited.clear();
    m_fadeOpacity.clear();

    for (const QString &path : paths) {
        QListWidgetItem *item = new QListWidgetItem(m_list);
        item->setData(Qt::UserRole, path);
        item->setIcon(QIcon());
    }

    if (m_list->count() > 0) m_list->setCurrentRow(0);
}

void GridView::setThumbnail(const QString &path, const QImage &thumb) {
    for (int i = 0; i < m_list->count(); ++i) {
        QListWidgetItem *item = m_list->item(i);
        if (item->data(Qt::UserRole).toString() == path) {
            item->setIcon(QIcon(QPixmap::fromImage(thumb)));
            break;
        }
    }
}

QImage GridView::thumbnail(const QString &path) const {
    for (int i = 0; i < m_list->count(); ++i) {
        const QListWidgetItem *item = m_list->item(i);
        if (item->data(Qt::UserRole).toString() == path) return item->icon().pixmap(m_list->iconSize()).toImage();
    }
    return {};
}

bool GridView::setCurrentPath(const QString &path) {
    for (int i = 0; i < m_list->count(); ++i) {
        QListWidgetItem *item = m_list->item(i);
        if (item->data(Qt::UserRole).toString() == path) {
            m_list->setCurrentItem(item);
            m_list->scrollToItem(item);
            return true;
        }
    }
    return false;
}

QString GridView::currentPath() const {
    const QListWidgetItem *item = m_list->currentItem();
    return item ? item->data(Qt::UserRole).toString() : QString();
}

void GridView::setMark(const QString &path, Mark m) {
    m_marks[path] = m;

    for (int i = 0; i < m_list->count(); ++i) {
        QListWidgetItem *item = m_list->item(i);
        if (item->data(Qt::UserRole).toString() == path) {
            switch (m) {
            case Mark::Accept:
                item->setBackground(QColor(144, 238, 144, 100));
                break;
            case Mark::Refine:
                item->setBackground(QColor(240, 210, 120, 110));
                break;
            case Mark::Decline:
                item->setBackground(QColor(255, 127, 127, 100));
                break;
            case Mark::None:
                item->setBackground(QColor(255, 255, 255, 0));
                break;
            }
            break;
        }
    }
}

GridView::Mark GridView::mark(const QString &path) const {
    return m_marks.value(path, Mark::None);
}

void GridView::toggleMark(const QString &path, Mark requested) {
    const Mark next = (mark(path) == requested) ? Mark::None : requested;
    setMark(path, next);
    emit markChanged(path, next);
}

void GridView::setProofStatus(const QString &path, ProofStatus status) {
    const ProofStatus prev = m_proofStatus.value(path, ProofStatus::NotProofed);
    m_proofStatus[path]    = status;

    if (status == ProofStatus::Proofing) {
        ensureAnimTimer();
    } else if (status == ProofStatus::Proofed && prev != ProofStatus::Proofed) {
        m_fadeOpacity[path] = 1.0f;
        ensureAnimTimer();
    }

    for (int i = 0; i < m_list->count(); ++i) {
        QListWidgetItem *item = m_list->item(i);
        if (item->data(Qt::UserRole).toString() == path) {
            m_list->update(m_list->indexFromItem(item));
            break;
        }
    }
}

void GridView::setEdited(const QString &path, bool edited) {
    m_edited[path] = edited;
    for (int i = 0; i < m_list->count(); ++i) {
        QListWidgetItem *item = m_list->item(i);
        if (item->data(Qt::UserRole).toString() == path) {
            m_list->update(m_list->indexFromItem(item));
            break;
        }
    }
}

bool GridView::isEdited(const QString &path) const {
    return m_edited.value(path, false);
}

void GridView::applyIconSize() {
    m_list->setIconSize(QSize(m_iconPx, m_iconPx));
    m_list->setGridSize(QSize(m_iconPx + kCardPadding * 2, m_iconPx + kCardPadding * 2));
    m_list->viewport()->update();
}

void GridView::ensureAnimTimer() {
    if (!m_animTimer) {
        m_animTimer = new QTimer(this);
        m_animTimer->setInterval(33); // ~30 fps
        connect(m_animTimer, &QTimer::timeout, this, &GridView::onAnimTick);
    }
    if (!m_animTimer->isActive()) m_animTimer->start();
}

void GridView::onAnimTick() {
    m_pulsePhase = std::fmod(m_pulsePhase + 0.025f, 1.0f); // ~1.3 s per cycle

    for (auto it = m_fadeOpacity.begin(); it != m_fadeOpacity.end();) {
        it.value() -= 0.04f; // fade over ~25 ticks ≈ 825 ms
        if (it.value() <= 0.0f) it = m_fadeOpacity.erase(it);
        else ++it;
    }

    const bool hasProofing = std::any_of(m_proofStatus.cbegin(), m_proofStatus.cend(),
                                         [](ProofStatus s) { return s == ProofStatus::Proofing; });

    if (!hasProofing && m_fadeOpacity.isEmpty()) m_animTimer->stop();

    m_list->viewport()->update();
}

void GridView::wheelEvent(QWheelEvent *event) {
    if (event->modifiers() & Qt::ControlModifier) {
        resizeThumbnails(event);
    } else {
        QWidget::wheelEvent(event);
    }
}

bool GridView::eventFilter(QObject *watched, QEvent *event) {
    if (watched == m_list->viewport() && event->type() == QEvent::Wheel) {
        auto *wheel = static_cast<QWheelEvent *>(event);
        if (wheel->modifiers() & Qt::ControlModifier) {
            resizeThumbnails(wheel);
            return true;
        }
    }
    return QWidget::eventFilter(watched, event);
}

void GridView::resizeThumbnails(QWheelEvent *event) {
    const int direction = event->angleDelta().y() > 0 ? 1 : (event->angleDelta().y() < 0 ? -1 : 0);
    if (direction != 0) {
        m_iconPx = qBound(64, m_iconPx + direction * 16, 512);
        applyIconSize();
    }
    event->accept();
}

void GridView::keyPressEvent(QKeyEvent *event) {
    QListWidgetItem *currentItem = m_list->currentItem();
    if (!currentItem) {
        QWidget::keyPressEvent(event);
        return;
    }

    const QString currentPath = currentItem->data(Qt::UserRole).toString();

    auto applyMark = [&](Mark requested) {
        toggleMark(currentPath, requested);
        m_list->setCurrentRow(m_list->currentRow() + 1);
        event->accept();
    };

    if (event->key() == Qt::Key_A) applyMark(Mark::Accept);
    else if (event->key() == Qt::Key_R) applyMark(Mark::Refine);
    else if (event->key() == Qt::Key_D) applyMark(Mark::Decline);
    else if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) emit photoActivated(currentPath);
    else QWidget::keyPressEvent(event);
}
