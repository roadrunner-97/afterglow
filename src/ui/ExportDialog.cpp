#include "ExportDialog.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSettings>
#include <QSlider>
#include <QVBoxLayout>

#include "Stylesheets.h"

namespace {
constexpr const char* kKeyDir               = "export/destinationDir";
constexpr const char* kKeyPattern           = "export/filenamePattern";
constexpr const char* kKeySubfolder         = "export/subfolder";
constexpr const char* kKeySubfolderExpanded = "export/subfolderExpanded";
constexpr const char* kKeyFormat            = "export/format";
constexpr const char* kKeyQuality           = "export/jpegQuality";
constexpr const char* kKeyConflict          = "export/onConflict";
} // namespace

ExportDialog::ExportDialog(QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle("Export Image");
    setMinimumWidth(420);

    auto* root = new QVBoxLayout(this);

    auto* form = new QFormLayout();
    form->setLabelAlignment(Qt::AlignRight);

    // ── Destination folder ────────────────────────────────────────────────
    {
        auto* row = new QHBoxLayout();
        m_destEdit  = new QLineEdit();
        m_browseBtn = new QPushButton("Browse…");
        row->addWidget(m_destEdit, 1);
        row->addWidget(m_browseBtn);
        form->addRow("Destination folder:", row);
        connect(m_browseBtn, &QPushButton::clicked,
                this, &ExportDialog::browseForDirectory);
    }

    // ── Filename pattern ──────────────────────────────────────────────────
    {
        m_patternEdit = new QLineEdit();
        form->addRow("Filename pattern:", m_patternEdit);
        auto* hint = new QLabel(
            "Tokens: <code>{name}</code> · <code>{n}</code> · <code>{date}</code>"
            " — anything else passes through verbatim.");
        hint->setTextFormat(Qt::RichText);
        hint->setWordWrap(true);
        // Spacer column under the label keeps the hint flush with the field.
        form->addRow(QString(), hint);
    }

    // ── Subfolder (collapsed by default) ──────────────────────────────────
    {
        auto* hdr = new QHBoxLayout();
        hdr->setContentsMargins(0, 0, 0, 0);
        m_subfolderToggle = new QPushButton("+");
        m_subfolderToggle->setStyleSheet(Stylesheets::collapseButton());
        m_subfolderToggle->setToolTip("Show or hide the subfolder field.");
        m_subfolderToggle->setMaximumWidth(28);
        hdr->addStretch();
        hdr->addWidget(m_subfolderToggle);
        form->addRow("Subfolder:", hdr);

        m_subfolderBody = new QWidget();
        auto* body = new QVBoxLayout(m_subfolderBody);
        body->setContentsMargins(0, 0, 0, 0);
        m_subfolderEdit = new QLineEdit();
        m_subfolderEdit->setPlaceholderText("optional — created if it doesn't exist");
        body->addWidget(m_subfolderEdit);
        auto* sfHint = new QLabel(
            "Nested paths allowed (e.g. <code>2026/exports</code>). "
            "Same tokens as the filename pattern are resolved here too.");
        sfHint->setTextFormat(Qt::RichText);
        sfHint->setWordWrap(true);
        body->addWidget(sfHint);
        form->addRow(QString(), m_subfolderBody);
        m_subfolderBody->setVisible(false);

        connect(m_subfolderToggle, &QPushButton::clicked, this, [this]() {
            const bool v = !m_subfolderBody->isVisible();
            m_subfolderBody->setVisible(v);
            m_subfolderToggle->setText(v ? "−" : "+");
        });
    }

    // ── Format ────────────────────────────────────────────────────────────
    m_formatCombo = new QComboBox();
    m_formatCombo->addItem("JPEG", static_cast<int>(ExportOptions::Format::JPEG));
    m_formatCombo->addItem("PNG",  static_cast<int>(ExportOptions::Format::PNG));
    m_formatCombo->addItem("TIFF", static_cast<int>(ExportOptions::Format::TIFF));
    form->addRow("Format:", m_formatCombo);

    // ── JPEG quality ──────────────────────────────────────────────────────
    {
        auto* row = new QHBoxLayout();
        m_qualitySlider = new QSlider(Qt::Horizontal);
        m_qualitySlider->setRange(1, 100);
        m_qualityLabel  = new QLabel("90");
        m_qualityLabel->setMinimumWidth(fontMetrics().horizontalAdvance("100"));
        row->addWidget(m_qualitySlider, 1);
        row->addWidget(m_qualityLabel);
        form->addRow("JPEG quality:", row);
        connect(m_qualitySlider, &QSlider::valueChanged, this, [this](int v) {
            m_qualityLabel->setText(QString::number(v));
        });
    }

    // ── Conflict policy ───────────────────────────────────────────────────
    m_conflictCombo = new QComboBox();
    m_conflictCombo->addItem("Append suffix (_1, _2, …)",
        static_cast<int>(ExportOptions::OverwritePolicy::AppendSuffix));
    m_conflictCombo->addItem("Skip",
        static_cast<int>(ExportOptions::OverwritePolicy::Skip));
    m_conflictCombo->addItem("Overwrite",
        static_cast<int>(ExportOptions::OverwritePolicy::Overwrite));
    form->addRow("When file exists:", m_conflictCombo);

    root->addLayout(form);

    // ── OK / Cancel ───────────────────────────────────────────────────────
    auto* buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    buttons->button(QDialogButtonBox::Ok)->setText("Export");
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    root->addWidget(buttons);

    connect(m_formatCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ExportDialog::onFormatChanged);

    loadFromSettings();
    onFormatChanged(m_formatCombo->currentIndex());
}

void ExportDialog::setDefaultDestinationDir(const QString& dir) {
    if (m_destEdit->text().isEmpty())
        m_destEdit->setText(dir);
}

ExportOptions::Options ExportDialog::options() const {
    ExportOptions::Options opts;
    opts.destinationDir  = m_destEdit->text();
    opts.subfolder       = m_subfolderEdit->text();
    opts.filenamePattern = m_patternEdit->text();
    opts.format          = static_cast<ExportOptions::Format>(
        m_formatCombo->currentData().toInt());
    opts.jpegQuality     = m_qualitySlider->value();
    opts.onConflict      = static_cast<ExportOptions::OverwritePolicy>(
        m_conflictCombo->currentData().toInt());
    return opts;
}

void ExportDialog::browseForDirectory() {
    const QString seed = m_destEdit->text();
    const QString chosen = QFileDialog::getExistingDirectory(
        this, "Choose Export Folder", seed);
    if (!chosen.isEmpty())
        m_destEdit->setText(chosen);
}

void ExportDialog::onFormatChanged(int /*idx*/) {
    const auto fmt = static_cast<ExportOptions::Format>(
        m_formatCombo->currentData().toInt());
    const bool jpeg = (fmt == ExportOptions::Format::JPEG);
    m_qualitySlider->setEnabled(jpeg);
    m_qualityLabel ->setEnabled(jpeg);
}

void ExportDialog::loadFromSettings() {
    QSettings s("Afterglow", "Afterglow");
    m_destEdit     ->setText(s.value(kKeyDir,       QString()).toString());
    m_patternEdit  ->setText(s.value(kKeyPattern,   "{name}").toString());
    m_subfolderEdit->setText(s.value(kKeySubfolder, QString()).toString());

    // Auto-expand if there's content to show, even if the user collapsed it
    // last time — otherwise a non-empty subfolder silently applies, which
    // surprises (cf. the original concern that drove this whole field).
    const bool hasSubfolder = !m_subfolderEdit->text().isEmpty();
    const bool expanded = hasSubfolder ||
        s.value(kKeySubfolderExpanded, false).toBool();
    m_subfolderBody  ->setVisible(expanded);
    m_subfolderToggle->setText(expanded ? "−" : "+");

    const int fmt      = s.value(kKeyFormat,
        static_cast<int>(ExportOptions::Format::JPEG)).toInt();
    const int quality  = s.value(kKeyQuality,  90).toInt();
    const int conflict = s.value(kKeyConflict,
        static_cast<int>(ExportOptions::OverwritePolicy::AppendSuffix)).toInt();

    if (const int idx = m_formatCombo  ->findData(fmt);      idx >= 0) m_formatCombo  ->setCurrentIndex(idx);
    if (const int idx = m_conflictCombo->findData(conflict); idx >= 0) m_conflictCombo->setCurrentIndex(idx);
    m_qualitySlider->setValue(quality);
    m_qualityLabel ->setText(QString::number(quality));
}

void ExportDialog::persistToSettings() const {
    QSettings s("Afterglow", "Afterglow");
    s.setValue(kKeyDir,               m_destEdit->text());
    s.setValue(kKeyPattern,           m_patternEdit->text());
    s.setValue(kKeySubfolder,         m_subfolderEdit->text());
    s.setValue(kKeySubfolderExpanded, m_subfolderBody->isVisible());
    s.setValue(kKeyFormat,            m_formatCombo->currentData().toInt());
    s.setValue(kKeyQuality,           m_qualitySlider->value());
    s.setValue(kKeyConflict,          m_conflictCombo->currentData().toInt());
}

void ExportDialog::accept() {
    persistToSettings();
    QDialog::accept();
}
