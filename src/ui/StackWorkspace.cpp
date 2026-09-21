#include "StackWorkspace.h"

#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QResizeEvent>
#include <QSignalBlocker>
#include <QSet>
#include <QVBoxLayout>

namespace {
constexpr int PATH_ROLE = Qt::UserRole;
}

StackWorkspace::StackWorkspace(QWidget *parent) : QWidget(parent) {
    setObjectName("stackWorkspace");
    auto *root = new QHBoxLayout(this);
    root->setContentsMargins(8, 8, 8, 8);
    root->setSpacing(8);

    auto *left = new QWidget();
    left->setMinimumWidth(260);
    left->setMaximumWidth(440);
    auto *leftLayout = new QVBoxLayout(left);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->addWidget(new QLabel("<b>Stack frames</b>"));
    auto *help = new QLabel("Checked frames contribute to the result. The golden reference supplies one shared set "
                            "of Develop adjustments to every frame.");
    help->setWordWrap(true);
    leftLayout->addWidget(help);

    m_frames = new QListWidget();
    m_frames->setObjectName("stackFrameList");
    m_frames->setSelectionMode(QAbstractItemView::ExtendedSelection);
    leftLayout->addWidget(m_frames, 1);
    m_frameSummary = new QLabel("0 included · 0 excluded");
    m_frameSummary->setObjectName("stackFrameSummary");
    leftLayout->addWidget(m_frameSummary);

    auto *add = new QPushButton("Add Photos…");
    add->setObjectName("addStackFramesButton");
    leftLayout->addWidget(add);
    auto *addFolderRaws = new QPushButton("Add Current Folder RAWs");
    addFolderRaws->setObjectName("addCurrentFolderRawsButton");
    addFolderRaws->setToolTip("Add every RAW photo from the folder currently open in Gallery.");
    leftLayout->addWidget(addFolderRaws);
    auto *selectionButtons = new QHBoxLayout();
    auto *includeAll       = new QPushButton("Include All");
    auto *excludeSelected  = new QPushButton("Exclude Selected");
    includeAll->setObjectName("includeAllStackFramesButton");
    excludeSelected->setObjectName("excludeSelectedStackFramesButton");
    selectionButtons->addWidget(includeAll);
    selectionButtons->addWidget(excludeSelected);
    leftLayout->addLayout(selectionButtons);

    m_setReference = new QPushButton("Set Selected as Reference");
    m_setReference->setObjectName("setStackReferenceButton");
    m_editReference = new QPushButton("Edit Reference in Develop");
    m_editReference->setObjectName("editStackReferenceButton");
    leftLayout->addWidget(m_setReference);
    leftLayout->addWidget(m_editReference);
    m_referenceLabel = new QLabel("Reference: none");
    m_referenceLabel->setObjectName("stackReferenceLabel");
    m_referenceLabel->setWordWrap(true);
    leftLayout->addWidget(m_referenceLabel);
    root->addWidget(left);

    auto *right       = new QWidget();
    auto *rightLayout = new QVBoxLayout(right);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    m_preview = new QLabel("Add photos to begin a long-exposure stack.");
    m_preview->setObjectName("stackPreview");
    m_preview->setAlignment(Qt::AlignCenter);
    m_preview->setFrameShape(QFrame::StyledPanel);
    m_preview->setMinimumSize(320, 240);
    m_preview->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    rightLayout->addWidget(m_preview, 1);

    m_status = new QLabel("The stack is rebuilt only when you ask, so Develop adjustments stay responsive.");
    m_status->setObjectName("stackStatusLabel");
    m_status->setWordWrap(true);
    rightLayout->addWidget(m_status);
    m_progress = new QProgressBar();
    m_progress->setObjectName("stackProgressBar");
    m_progress->setVisible(false);
    rightLayout->addWidget(m_progress);

    auto *actions = new QHBoxLayout();
    m_rebuild     = new QPushButton("Rebuild Stack");
    m_rebuild->setObjectName("rebuildStackButton");
    m_cancel = new QPushButton("Cancel");
    m_cancel->setObjectName("cancelStackButton");
    m_save = new QPushButton("Save Stack…");
    m_save->setObjectName("saveStackButton");
    m_cancel->setEnabled(false);
    m_save->setEnabled(false);
    actions->addWidget(m_rebuild);
    actions->addWidget(m_cancel);
    actions->addStretch();
    actions->addWidget(m_save);
    rightLayout->addLayout(actions);
    root->addWidget(right, 1);

    connect(add, &QPushButton::clicked, this, &StackWorkspace::addFramesRequested);
    connect(addFolderRaws, &QPushButton::clicked, this, &StackWorkspace::addCurrentFolderRawsRequested);
    connect(m_rebuild, &QPushButton::clicked, this, &StackWorkspace::rebuildRequested);
    connect(m_cancel, &QPushButton::clicked, this, &StackWorkspace::cancelRequested);
    connect(m_save, &QPushButton::clicked, this, &StackWorkspace::saveRequested);
    connect(m_frames, &QListWidget::itemChanged, this, [this](QListWidgetItem *) { updateFrameSummary(); });
    connect(m_frames, &QListWidget::itemSelectionChanged, this,
            [this]() { m_setReference->setEnabled(m_frames->selectedItems().size() == 1); });
    connect(includeAll, &QPushButton::clicked, this, [this]() {
        for (int i = 0; i < m_frames->count(); ++i) m_frames->item(i)->setCheckState(Qt::Checked);
    });
    connect(excludeSelected, &QPushButton::clicked, this, [this]() {
        for (QListWidgetItem *item : m_frames->selectedItems()) item->setCheckState(Qt::Unchecked);
    });
    connect(m_setReference, &QPushButton::clicked, this, [this]() {
        const auto selected = m_frames->selectedItems();
        if (selected.size() == 1) setReferencePath(selected.first()->data(PATH_ROLE).toString());
    });
    connect(m_editReference, &QPushButton::clicked, this, [this]() {
        if (!m_referencePath.isEmpty()) emit editReferenceRequested(m_referencePath);
    });

    m_setReference->setEnabled(false);
    m_editReference->setEnabled(false);
    m_rebuild->setEnabled(false);
}

int StackWorkspace::addFrames(const QStringList &paths) {
    QSet<QString> existing;
    for (int i = 0; i < m_frames->count(); ++i) existing.insert(m_frames->item(i)->data(PATH_ROLE).toString());

    int                  added = 0;
    const QSignalBlocker blocker(m_frames);
    for (const QString &path : paths) {
        const QString absolute = QFileInfo(path).absoluteFilePath();
        if (absolute.isEmpty() || existing.contains(absolute)) continue;
        auto *item = new QListWidgetItem(QFileInfo(absolute).fileName(), m_frames);
        item->setData(PATH_ROLE, absolute);
        item->setToolTip(absolute);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked);
        existing.insert(absolute);
        ++added;
    }
    if (m_referencePath.isEmpty() && m_frames->count() > 0)
        m_referencePath = m_frames->item(0)->data(PATH_ROLE).toString();
    updateReferencePresentation();
    updateFrameSummary();
    return added;
}

QVector<StackFrame> StackWorkspace::frames() const {
    QVector<StackFrame> result;
    result.reserve(m_frames->count());
    for (int i = 0; i < m_frames->count(); ++i) {
        const QListWidgetItem *item = m_frames->item(i);
        result.append({item->data(PATH_ROLE).toString(),
                       item->checkState() == Qt::Checked ? StackFrameDecision::Include : StackFrameDecision::Exclude});
    }
    return result;
}

QString StackWorkspace::referencePath() const {
    return m_referencePath;
}

void StackWorkspace::setReferencePath(const QString &path) {
    for (int i = 0; i < m_frames->count(); ++i) {
        if (m_frames->item(i)->data(PATH_ROLE).toString() != path) continue;
        m_referencePath = path;
        updateReferencePresentation();
        return;
    }
}

void StackWorkspace::setResult(const QImage &result) {
    m_result = result;
    m_save->setEnabled(!result.isNull());
    updatePreviewPixmap();
}

QImage StackWorkspace::result() const {
    return m_result;
}

void StackWorkspace::setBuilding(bool building, int totalFrames) {
    m_rebuild->setEnabled(!building && !frames().isEmpty());
    m_cancel->setEnabled(building);
    m_progress->setVisible(building);
    if (building) {
        m_progress->setRange(0, totalFrames);
        m_progress->setValue(0);
        m_status->setText(QStringLiteral("Rendering 0 of %1 frames…").arg(totalFrames));
    } else m_progress->setVisible(false);
}

void StackWorkspace::setProgress(int completedFrames, int totalFrames, const QString &path) {
    m_progress->setRange(0, totalFrames);
    m_progress->setValue(completedFrames);
    m_status->setText(
        QStringLiteral("Rendered %1 of %2 · %3").arg(completedFrames).arg(totalFrames).arg(QFileInfo(path).fileName()));
}

void StackWorkspace::setStatus(const QString &message) {
    m_status->setText(message);
}

void StackWorkspace::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
    updatePreviewPixmap();
}

void StackWorkspace::updateFrameSummary() {
    int included = 0;
    for (int i = 0; i < m_frames->count(); ++i)
        if (m_frames->item(i)->checkState() == Qt::Checked) ++included;
    m_frameSummary->setText(
        QStringLiteral("%1 included · %2 excluded").arg(included).arg(m_frames->count() - included));
    if (!m_cancel->isEnabled()) m_rebuild->setEnabled(included > 0 && !m_referencePath.isEmpty());
}

void StackWorkspace::updateReferencePresentation() {
    for (int i = 0; i < m_frames->count(); ++i) {
        QListWidgetItem *item      = m_frames->item(i);
        const bool       reference = item->data(PATH_ROLE).toString() == m_referencePath;
        QFont            font      = item->font();
        font.setBold(reference);
        item->setFont(font);
        item->setText((reference ? QStringLiteral("★ ") : QString()) +
                      QFileInfo(item->data(PATH_ROLE).toString()).fileName());
    }
    m_referenceLabel->setText(m_referencePath.isEmpty()
                                  ? QStringLiteral("Reference: none")
                                  : QStringLiteral("Reference: %1").arg(QFileInfo(m_referencePath).fileName()));
    m_editReference->setEnabled(!m_referencePath.isEmpty());
    updateFrameSummary();
}

void StackWorkspace::updatePreviewPixmap() {
    if (m_result.isNull()) {
        m_preview->setPixmap({});
        m_preview->setText("Add photos to begin a long-exposure stack.");
        return;
    }
    m_preview->setText({});
    m_preview->setPixmap(QPixmap::fromImage(m_result).scaled(m_preview->size() - QSize(12, 12), Qt::KeepAspectRatio,
                                                             Qt::SmoothTransformation));
}
