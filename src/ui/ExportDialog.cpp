#include "ExportDialog.h"

#include <QCheckBox>
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
#include <QSpinBox>
#include <QVBoxLayout>

namespace {
constexpr const char *kKeyDir               = "export/destinationDir";
constexpr const char *kKeyPattern           = "export/filenamePattern";
constexpr const char *kKeySubfolder         = "export/subfolder";
constexpr const char *kKeySubfolderExpanded = "export/subfolderExpanded";
constexpr const char *kKeyFormat            = "export/format";
constexpr const char *kKeyQuality           = "export/jpegQuality";
constexpr const char *kKeyConflict          = "export/onConflict";
constexpr const char *kKeyResizeMode        = "export/resizeMode";
constexpr const char *kKeyResizePixels      = "export/resizePixels";
constexpr const char *kKeyResizePercent     = "export/resizePercent";
constexpr const char *kKeyResizeNoEnlarge   = "export/resizeNoEnlarge";
} // namespace

ExportDialog::ExportDialog(QWidget *parent) : QDialog(parent) {
    setWindowTitle("Export Image");
    setMinimumWidth(420);

    auto *root = new QVBoxLayout(this);

    auto *form = new QFormLayout();
    form->setLabelAlignment(Qt::AlignRight);

    // ── Destination folder ────────────────────────────────────────────────
    {
        auto *row   = new QHBoxLayout();
        m_destEdit  = new QLineEdit();
        m_browseBtn = new QPushButton("Browse…");
        row->addWidget(m_destEdit, 1);
        row->addWidget(m_browseBtn);
        form->addRow("Destination folder:", row);
        connect(m_browseBtn, &QPushButton::clicked, this, &ExportDialog::browseForDirectory);
    }

    // ── Filename pattern ──────────────────────────────────────────────────
    {
        m_patternEdit = new QLineEdit();
        form->addRow("Filename pattern:", m_patternEdit);
        auto *hint = new QLabel("Tokens: <code>{name}</code> · <code>{n}</code> · <code>{date}</code>"
                                " — anything else passes through verbatim.");
        hint->setTextFormat(Qt::RichText);
        hint->setWordWrap(true);
        // Spacer column under the label keeps the hint flush with the field.
        form->addRow(QString(), hint);
    }

    // ── Subfolder (collapsed by default) ──────────────────────────────────
    {
        auto *hdr = new QHBoxLayout();
        hdr->setContentsMargins(0, 0, 0, 0);
        m_subfolderToggle = new QPushButton("+");
        m_subfolderToggle->setToolTip("Show or hide the subfolder field.");
        m_subfolderToggle->setMaximumWidth(28);
        hdr->addStretch();
        hdr->addWidget(m_subfolderToggle);
        form->addRow("Subfolder:", hdr);

        m_subfolderBody = new QWidget();
        auto *body      = new QVBoxLayout(m_subfolderBody);
        body->setContentsMargins(0, 0, 0, 0);
        m_subfolderEdit = new QLineEdit();
        m_subfolderEdit->setPlaceholderText("optional — created if it doesn't exist");
        body->addWidget(m_subfolderEdit);
        auto *sfHint = new QLabel("Nested paths allowed (e.g. <code>2026/exports</code>). "
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
    m_formatCombo->addItem("PNG", static_cast<int>(ExportOptions::Format::PNG));
    m_formatCombo->addItem("TIFF", static_cast<int>(ExportOptions::Format::TIFF));
    form->addRow("Format:", m_formatCombo);

    // ── JPEG quality ──────────────────────────────────────────────────────
    {
        auto *row       = new QHBoxLayout();
        m_qualitySlider = new QSlider(Qt::Horizontal);
        m_qualitySlider->setRange(1, 100);
        m_qualityLabel = new QLabel("90");
        m_qualityLabel->setMinimumWidth(fontMetrics().horizontalAdvance("100"));
        row->addWidget(m_qualitySlider, 1);
        row->addWidget(m_qualityLabel);
        form->addRow("JPEG quality:", row);
        connect(m_qualitySlider, &QSlider::valueChanged, this,
                [this](int v) { m_qualityLabel->setText(QString::number(v)); });
    }

    // ── Conflict policy ───────────────────────────────────────────────────
    m_conflictCombo = new QComboBox();
    m_conflictCombo->addItem("Append suffix (_1, _2, …)",
                             static_cast<int>(ExportOptions::OverwritePolicy::AppendSuffix));
    m_conflictCombo->addItem("Skip", static_cast<int>(ExportOptions::OverwritePolicy::Skip));
    m_conflictCombo->addItem("Overwrite", static_cast<int>(ExportOptions::OverwritePolicy::Overwrite));
    form->addRow("When file exists:", m_conflictCombo);

    // ── Resize ────────────────────────────────────────────────────────────
    {
        m_resizeModeCombo = new QComboBox();
        m_resizeModeCombo->addItem("Original size", static_cast<int>(ExportResize::Mode::None));
        m_resizeModeCombo->addItem("Long edge", static_cast<int>(ExportResize::Mode::LongEdge));
        m_resizeModeCombo->addItem("Short edge", static_cast<int>(ExportResize::Mode::ShortEdge));
        m_resizeModeCombo->addItem("Width", static_cast<int>(ExportResize::Mode::Width));
        m_resizeModeCombo->addItem("Height", static_cast<int>(ExportResize::Mode::Height));
        m_resizeModeCombo->addItem("Percentage", static_cast<int>(ExportResize::Mode::Percentage));
        form->addRow("Resize:", m_resizeModeCombo);

        auto *row          = new QHBoxLayout();
        m_resizePixelsSpin = new QSpinBox();
        m_resizePixelsSpin->setRange(1, 32768);
        m_resizePixelsSpin->setSuffix(" px");
        m_resizePercentSpin = new QSpinBox();
        m_resizePercentSpin->setRange(1, 1000);
        m_resizePercentSpin->setSuffix(" %");
        row->addWidget(m_resizePixelsSpin);
        row->addWidget(m_resizePercentSpin);
        row->addStretch();
        form->addRow(QString(), row);

        m_resizeNoEnlarge = new QCheckBox("Don't enlarge");
        form->addRow(QString(), m_resizeNoEnlarge);

        connect(m_resizeModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                &ExportDialog::onResizeModeChanged);
    }

    root->addLayout(form);

    // ── OK / Cancel ───────────────────────────────────────────────────────
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    buttons->button(QDialogButtonBox::Ok)->setText("Export");
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    root->addWidget(buttons);

    connect(m_formatCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ExportDialog::onFormatChanged);

    loadFromSettings();
    onFormatChanged(m_formatCombo->currentIndex());
}

void ExportDialog::setDefaultDestinationDir(const QString &dir) {
    if (m_destEdit->text().isEmpty()) m_destEdit->setText(dir);
}

ExportOptions::Options ExportDialog::options() const {
    ExportOptions::Options opts;
    opts.destinationDir     = m_destEdit->text();
    opts.subfolder          = m_subfolderEdit->text();
    opts.filenamePattern    = m_patternEdit->text();
    opts.format             = static_cast<ExportOptions::Format>(m_formatCombo->currentData().toInt());
    opts.jpegQuality        = m_qualitySlider->value();
    opts.onConflict         = static_cast<ExportOptions::OverwritePolicy>(m_conflictCombo->currentData().toInt());
    opts.resize.mode        = static_cast<ExportResize::Mode>(m_resizeModeCombo->currentData().toInt());
    opts.resize.pixels      = m_resizePixelsSpin->value();
    opts.resize.percent     = m_resizePercentSpin->value();
    opts.resize.dontEnlarge = m_resizeNoEnlarge->isChecked();
    return opts;
}

void ExportDialog::browseForDirectory() {
    const QString seed   = m_destEdit->text();
    const QString chosen = QFileDialog::getExistingDirectory(this, "Choose Export Folder", seed);
    if (!chosen.isEmpty()) m_destEdit->setText(chosen);
}

void ExportDialog::onFormatChanged(int /*idx*/) {
    const auto fmt  = static_cast<ExportOptions::Format>(m_formatCombo->currentData().toInt());
    const bool jpeg = (fmt == ExportOptions::Format::JPEG);
    m_qualitySlider->setEnabled(jpeg);
    m_qualityLabel->setEnabled(jpeg);
}

void ExportDialog::onResizeModeChanged(int /*idx*/) {
    const auto mode    = static_cast<ExportResize::Mode>(m_resizeModeCombo->currentData().toInt());
    const bool active  = (mode != ExportResize::Mode::None);
    const bool percent = (mode == ExportResize::Mode::Percentage);
    m_resizePixelsSpin->setVisible(active && !percent);
    m_resizePercentSpin->setVisible(active && percent);
    m_resizeNoEnlarge->setEnabled(active);
}

void ExportDialog::loadFromSettings() {
    QSettings s("Afterglow", "Afterglow");
    m_destEdit->setText(s.value(kKeyDir, QString()).toString());
    m_patternEdit->setText(s.value(kKeyPattern, "{name}").toString());
    m_subfolderEdit->setText(s.value(kKeySubfolder, QString()).toString());

    // Auto-expand if there's content to show, even if the user collapsed it
    // last time — otherwise a non-empty subfolder silently applies, which
    // surprises (cf. the original concern that drove this whole field).
    const bool hasSubfolder = !m_subfolderEdit->text().isEmpty();
    const bool expanded     = hasSubfolder || s.value(kKeySubfolderExpanded, false).toBool();
    m_subfolderBody->setVisible(expanded);
    m_subfolderToggle->setText(expanded ? "−" : "+");

    const int fmt      = s.value(kKeyFormat, static_cast<int>(ExportOptions::Format::JPEG)).toInt();
    const int quality  = s.value(kKeyQuality, 90).toInt();
    const int conflict = s.value(kKeyConflict, static_cast<int>(ExportOptions::OverwritePolicy::AppendSuffix)).toInt();

    if (const int idx = m_formatCombo->findData(fmt); idx >= 0) m_formatCombo->setCurrentIndex(idx);
    if (const int idx = m_conflictCombo->findData(conflict); idx >= 0) m_conflictCombo->setCurrentIndex(idx);
    m_qualitySlider->setValue(quality);
    m_qualityLabel->setText(QString::number(quality));

    const int  rmode    = s.value(kKeyResizeMode, static_cast<int>(ExportResize::Mode::None)).toInt();
    const int  rpixels  = s.value(kKeyResizePixels, 2048).toInt();
    const int  rpercent = s.value(kKeyResizePercent, 100).toInt();
    const bool rnoenl   = s.value(kKeyResizeNoEnlarge, true).toBool();
    if (const int idx = m_resizeModeCombo->findData(rmode); idx >= 0) m_resizeModeCombo->setCurrentIndex(idx);
    m_resizePixelsSpin->setValue(rpixels);
    m_resizePercentSpin->setValue(rpercent);
    m_resizeNoEnlarge->setChecked(rnoenl);
    onResizeModeChanged(m_resizeModeCombo->currentIndex());
}

void ExportDialog::persistToSettings() const {
    QSettings s("Afterglow", "Afterglow");
    s.setValue(kKeyDir, m_destEdit->text());
    s.setValue(kKeyPattern, m_patternEdit->text());
    s.setValue(kKeySubfolder, m_subfolderEdit->text());
    s.setValue(kKeySubfolderExpanded, m_subfolderBody->isVisible());
    s.setValue(kKeyFormat, m_formatCombo->currentData().toInt());
    s.setValue(kKeyQuality, m_qualitySlider->value());
    s.setValue(kKeyConflict, m_conflictCombo->currentData().toInt());
    s.setValue(kKeyResizeMode, m_resizeModeCombo->currentData().toInt());
    s.setValue(kKeyResizePixels, m_resizePixelsSpin->value());
    s.setValue(kKeyResizePercent, m_resizePercentSpin->value());
    s.setValue(kKeyResizeNoEnlarge, m_resizeNoEnlarge->isChecked());
}

void ExportDialog::accept() {
    persistToSettings();
    QDialog::accept();
}
