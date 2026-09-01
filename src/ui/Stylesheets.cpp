#include "Stylesheets.h"

namespace Stylesheets {

QString toolbar() {
    return QString("QToolBar { background: palette(window); border-bottom: 1px solid palette(mid); spacing: 6px; "
                   "padding: 2px 6px; }"
                   "QToolButton { color: palette(window-text); background: transparent; border: 1px solid palette(mid);"
                   "  border-radius: 3px; padding: 3px 10px; }"
                   "QToolButton:checked { color: palette(highlighted-text); background: palette(highlight); "
                   "border-color: palette(highlight); }"
                   "QToolButton:hover { background: palette(button); }");
}

QString processingLabel() {
    return "color: palette(mid); font-style: italic; padding: 0 6px;";
}

QString menuBar() {
    return QString(
        "QMenuBar { background: palette(window); color: palette(window-text); border-bottom: 1px solid palette(mid); }"
        "QMenuBar::item { padding: 4px 8px; }"
        "QMenuBar::item:selected { background: palette(button); border-radius: 3px; }"
        "QMenu { background: palette(window); color: palette(window-text); border: 1px solid palette(mid); }"
        "QMenu::item { padding: 4px 20px; }"
        "QMenu::item:selected { background: palette(highlight); color: palette(highlighted-text); }"
        "QMenu::separator { height: 1px; background: palette(mid); margin: 2px 0; }");
}

QString gpuSelectorLabel() {
    return "color: palette(mid); text-transform: uppercase;";
}

QString gpuSelector() {
    return QString("QComboBox { color: palette(text); background-color: palette(base);"
                   "  border: 1px solid palette(mid); border-radius: 3px; padding: 4px; }"
                   "QComboBox::drop-down { border: none; }"
                   "QComboBox QAbstractItemView { color: palette(text); background-color: palette(base); }");
}

QString effectPanel() {
    return "QWidget { background-color: palette(base); border-radius: 4px; }";
}

QString effectTitle() {
    return "color: palette(text); background: transparent;";
}

QString collapseButton() {
    return QString("QPushButton { background: palette(button); color: palette(button-text); border: none;"
                   "  border-radius: 3px; padding: 1px 5px; font-weight: bold; }"
                   "QPushButton:hover { background: palette(mid); }");
}

QString panelSeparator() {
    return "color: palette(mid);";
}

} // namespace Stylesheets
