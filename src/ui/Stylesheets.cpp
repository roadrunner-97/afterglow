#include "Stylesheets.h"
#include "Theme.h"

namespace Stylesheets {

QString toolbar() {
    return QString(
        "QToolBar { background: %1; border-bottom: 1px solid %2; spacing: 6px; padding: 2px 6px; }"
        "QToolButton { color: %3; background: transparent; border: 1px solid %2;"
        "  border-radius: 3px; padding: 3px 10px; }"
        "QToolButton:checked { color: %4; background: %5; border-color: %5; }"
        "QToolButton:hover { background: %6; }")
        .arg(Theme::BG_MAIN, Theme::BORDER,
             Theme::TEXT_PRIMARY, Theme::CHECKED_TEXT,
             Theme::CHECKED_BG, Theme::COLLAPSE_HOVER);
}

QString processingLabel() {
    return QString("color: %1; font-style: italic; padding: 0 6px;")
        .arg(Theme::TEXT_SECONDARY);
}

QString menuBar() {
    return QString(
        "QMenuBar { background: %1; color: %2; border-bottom: 1px solid %3; }"
        "QMenuBar::item { padding: 4px 8px; }"
        "QMenuBar::item:selected { background: %4; border-radius: 3px; }"
        "QMenu { background: %5; color: %2; border: 1px solid %3; }"
        "QMenu::item { padding: 4px 20px; }"
        "QMenu::item:selected { background: %6; color: %7; }"
        "QMenu::separator { height: 1px; background: %3; margin: 2px 0; }")
        .arg(Theme::BG_MAIN, Theme::TEXT_PRIMARY, Theme::BORDER,
             Theme::COLLAPSE_HOVER, Theme::BG_EFFECT_PANEL,
             Theme::CHECKED_BG, Theme::CHECKED_TEXT);
}

QString gpuSelectorLabel() {
    return QString("color: %1; font-size: 10px; text-transform: uppercase;")
        .arg(Theme::TEXT_SECONDARY);
}

QString gpuSelector() {
    return QString(
        "QComboBox { color: %1; background-color: %2;"
        "  border: 1px solid %3; border-radius: 3px; padding: 4px; }"
        "QComboBox::drop-down { border: none; }"
        "QComboBox QAbstractItemView { color: %1; background-color: %2; }")
        .arg(Theme::TEXT_PRIMARY, Theme::BG_EFFECT_PANEL, Theme::BORDER);
}

QString effectPanel() {
    return QString("QWidget { background-color: %1; border-radius: 4px; }")
        .arg(Theme::BG_EFFECT_PANEL);
}

QString effectTitle() {
    return QString("color: %1; background: transparent;").arg(Theme::TEXT_PRIMARY);
}

QString collapseButton() {
    return QString(
        "QPushButton { background: %1; color: %2; border: none;"
        "  border-radius: 3px; padding: 1px 5px; font-weight: bold; }"
        "QPushButton:hover { background: %3; }")
        .arg(Theme::COLLAPSE_BG, Theme::TEXT_PRIMARY, Theme::COLLAPSE_HOVER);
}

QString panelSeparator() {
    return QString("color: %1;").arg(Theme::BORDER_PANEL);
}

} // namespace Stylesheets
