#include "GridView.h"

#include <QListWidget>
#include <QListWidgetItem>
#include <QPainter>
#include <QStyledItemDelegate>
#include <QStyleOptionViewItem>
#include <QTimer>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QFileInfo>
#include <QPixmap>

#include <algorithm>
#include <cmath>

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Draws a glowing colored ring around the thumbnail border.
// Red = NotProofed, orange-pulsing = Proofing, orange-fading = Proofed.
class ProofRingDelegate : public QStyledItemDelegate {
public:
    explicit ProofRingDelegate(const QHash<QString, GridView::ProofStatus> *status, const QHash<QString, bool> *edited,
                               const QHash<QString, float> *fadeOpacity, const float *pulsePhase,
                               QObject *parent = nullptr)
        : QStyledItemDelegate(parent), m_status(status), m_edited(edited), m_fadeOpacity(fadeOpacity),
          m_pulsePhase(pulsePhase) {}

    void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override {
        QStyledItemDelegate::paint(painter, option, index);

        const QString               path   = index.data(Qt::UserRole).toString();
        const GridView::ProofStatus status = m_status->value(path, GridView::ProofStatus::NotProofed);

        QColor baseColor;
        float  opacity  = 1.0f;
        bool   drawRing = true;

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
                drawRing = false;
                break;
            }
            baseColor = QColor(255, 155, 25);
            opacity   = fade;
            break;
        }
        }

        const QSize iconSz = option.decorationSize;
        const int   iLeft  = option.rect.left() + (option.rect.width() - iconSz.width()) / 2;
        const int   iTop   = option.rect.top() + (option.rect.height() - iconSz.height()) / 2;
        const QRect iconRect(iLeft, iTop, iconSz.width(), iconSz.height());

        painter->save();
        painter->setBrush(Qt::NoBrush);

        if (drawRing) {
            // Glow halos: 3 progressively wider, progressively more transparent rects
            const float glowAlphas[] = {0.35f, 0.18f, 0.08f};
            for (int i = 0; i < 3; ++i) {
                QColor g = baseColor;
                g.setAlphaF(opacity * glowAlphas[i]);
                painter->setPen(QPen(g, 1));
                const int d = i + 1;
                painter->drawRect(iconRect.adjusted(-d, -d, d, d));
            }

            // Main ring
            QColor ring = baseColor;
            ring.setAlphaF(opacity * 0.90f);
            painter->setPen(QPen(ring, 2));
            painter->drawRect(iconRect.adjusted(-1, -1, 1, 1));
        }

        if (m_edited->value(path, false)) {
            const QString text = QStringLiteral("Edited");
            QFont         font = painter->font();
            font.setBold(true);
            font.setPointSizeF(std::max(8.0, font.pointSizeF() - 1.0));
            painter->setFont(font);
            const QFontMetrics fm(font);
            const QSize        badgeSize(fm.horizontalAdvance(text) + 12, fm.height() + 4);
            const QRect badge(iconRect.right() - badgeSize.width() - 4, iconRect.bottom() - badgeSize.height() - 4,
                              badgeSize.width(), badgeSize.height());
            painter->setPen(Qt::NoPen);
            painter->setBrush(QColor(30, 30, 30, 210));
            painter->drawRoundedRect(badge, 4, 4);
            painter->setPen(Qt::white);
            painter->drawText(badge, Qt::AlignCenter, text);
        }

        painter->restore();
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
    m_list->setItemDelegate(new ProofRingDelegate(&m_proofStatus, &m_edited, &m_fadeOpacity, &m_pulsePhase, this));
    m_list->setViewMode(QListView::IconMode);
    m_list->setResizeMode(QListView::Adjust);
    m_list->setMovement(QListView::Static);
    m_list->setWrapping(true);
    m_list->setFocusPolicy(Qt::StrongFocus);
    m_list->setFocusProxy(m_list);

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
    m_list->setGridSize(QSize(m_iconPx + 16, m_iconPx + 16));
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
        int delta = event->angleDelta().y() / 8;
        m_iconPx  = qBound(48, m_iconPx + delta, 512);
        applyIconSize();
        event->accept();
    } else {
        QWidget::wheelEvent(event);
    }
}

void GridView::keyPressEvent(QKeyEvent *event) {
    QListWidgetItem *currentItem = m_list->currentItem();
    if (!currentItem) {
        QWidget::keyPressEvent(event);
        return;
    }

    const QString currentPath = currentItem->data(Qt::UserRole).toString();

    auto applyMark = [&](Mark requested) {
        const Mark next = (mark(currentPath) == requested) ? Mark::None : requested;
        setMark(currentPath, next);
        emit markChanged(currentPath, next);
        m_list->setCurrentRow(m_list->currentRow() + 1);
        event->accept();
    };

    if (event->key() == Qt::Key_A) applyMark(Mark::Accept);
    else if (event->key() == Qt::Key_R) applyMark(Mark::Refine);
    else if (event->key() == Qt::Key_D) applyMark(Mark::Decline);
    else if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) emit photoActivated(currentPath);
    else QWidget::keyPressEvent(event);
}
