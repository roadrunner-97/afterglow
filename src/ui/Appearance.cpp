#include "Appearance.h"

#include <QApplication>
#include <QFont>
#include <QSettings>

namespace {
QPalette initialPalette;
QFont    initialFont;
bool     captured = false;

void captureDefaults() {
    if (captured) return;
    initialPalette = QApplication::palette();
    initialFont    = QApplication::font();
    captured       = true;
}

QPalette lightPalette() {
    QPalette palette = initialPalette;
    palette.setColor(QPalette::Window, QColor("#F0EDE5"));
    palette.setColor(QPalette::WindowText, QColor("#2C2018"));
    palette.setColor(QPalette::Base, QColor("#F8F5F0"));
    palette.setColor(QPalette::AlternateBase, QColor("#EDEADE"));
    palette.setColor(QPalette::ToolTipBase, QColor("#F8F5F0"));
    palette.setColor(QPalette::ToolTipText, QColor("#2C2018"));
    palette.setColor(QPalette::Text, QColor("#2C2018"));
    palette.setColor(QPalette::Button, QColor("#E6E0D4"));
    palette.setColor(QPalette::ButtonText, QColor("#2C2018"));
    palette.setColor(QPalette::BrightText, Qt::white);
    palette.setColor(QPalette::Light, QColor("#FFFFFF"));
    palette.setColor(QPalette::Midlight, QColor("#E1DCCE"));
    palette.setColor(QPalette::Dark, QColor("#9C9485"));
    palette.setColor(QPalette::Shadow, QColor("#625A4F"));
    palette.setColor(QPalette::Link, QColor("#5B6EA8"));
    palette.setColor(QPalette::LinkVisited, QColor("#765B91"));
    palette.setColor(QPalette::Highlight, QColor("#C0802C"));
    palette.setColor(QPalette::HighlightedText, QColor("#F5F2EA"));
    palette.setColor(QPalette::Mid, QColor("#CCC5B5"));
    palette.setColor(QPalette::PlaceholderText, QColor("#766C60"));
    palette.setColor(QPalette::Disabled, QPalette::WindowText, QColor("#887E70"));
    palette.setColor(QPalette::Disabled, QPalette::Text, QColor("#887E70"));
    palette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor("#887E70"));
    return palette;
}

QPalette darkPalette() {
    QPalette palette = initialPalette;
    palette.setColor(QPalette::Window, QColor("#25272B"));
    palette.setColor(QPalette::WindowText, QColor("#ECEDEF"));
    palette.setColor(QPalette::Base, QColor("#1B1D20"));
    palette.setColor(QPalette::AlternateBase, QColor("#303238"));
    palette.setColor(QPalette::ToolTipBase, QColor("#35383E"));
    palette.setColor(QPalette::ToolTipText, QColor("#ECEDEF"));
    palette.setColor(QPalette::Text, QColor("#ECEDEF"));
    palette.setColor(QPalette::Button, QColor("#35383E"));
    palette.setColor(QPalette::ButtonText, QColor("#ECEDEF"));
    palette.setColor(QPalette::BrightText, Qt::white);
    palette.setColor(QPalette::Light, QColor("#62666E"));
    palette.setColor(QPalette::Midlight, QColor("#484B52"));
    palette.setColor(QPalette::Dark, QColor("#17181B"));
    palette.setColor(QPalette::Shadow, QColor("#090A0B"));
    palette.setColor(QPalette::Link, QColor("#8EACFF"));
    palette.setColor(QPalette::LinkVisited, QColor("#C2A2E8"));
    palette.setColor(QPalette::Highlight, QColor("#C98B3C"));
    palette.setColor(QPalette::HighlightedText, QColor("#151618"));
    palette.setColor(QPalette::Mid, QColor("#555960"));
    palette.setColor(QPalette::PlaceholderText, QColor("#A9ACB2"));
    palette.setColor(QPalette::Disabled, QPalette::WindowText, QColor("#8B8E94"));
    palette.setColor(QPalette::Disabled, QPalette::Text, QColor("#8B8E94"));
    palette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor("#8B8E94"));
    palette.setColor(QPalette::Disabled, QPalette::HighlightedText, QColor("#6F7278"));
    return palette;
}
} // namespace

namespace Appearance {

const QList<ThemeDefinition> &themes() {
    static const QList<ThemeDefinition> definitions = {{"system", "Follow system", {}, true},
                                                       {"light", "Light", lightPalette(), false},
                                                       {"dark", "Dark", darkPalette(), false}};
    return definitions;
}

QString defaultThemeId() {
    return "system";
}

QString savedThemeId() {
    return QSettings("Afterglow", "Afterglow").value("appearance/theme", defaultThemeId()).toString();
}

QString savedFontFamily() {
    captureDefaults();
    return QSettings("Afterglow", "Afterglow").value("appearance/fontFamily", initialFont.family()).toString();
}

int savedFontSize() {
    captureDefaults();
    const int fallback = initialFont.pointSize() > 0 ? initialFont.pointSize() : 10;
    return QSettings("Afterglow", "Afterglow").value("appearance/fontSize", fallback).toInt();
}

void initialize() {
    captureDefaults();
    apply(savedThemeId(), savedFontFamily(), savedFontSize());
}

void apply(const QString &themeId, const QString &fontFamily, int fontSize) {
    captureDefaults();
    auto selected = themes().cbegin();
    for (auto it = themes().cbegin(); it != themes().cend(); ++it) {
        if (it->id == themeId) {
            selected = it;
            break;
        }
    }
    QApplication::setPalette(selected->usesSystemPalette ? initialPalette : selected->palette);
    QFont font(fontFamily.isEmpty() ? initialFont.family() : fontFamily);
    font.setPointSize(qBound(8, fontSize, 24));
    QApplication::setFont(font);
}

void saveAndApply(const QString &themeId, const QString &fontFamily, int fontSize) {
    QSettings settings("Afterglow", "Afterglow");
    settings.setValue("appearance/theme", themeId);
    settings.setValue("appearance/fontFamily", fontFamily);
    settings.setValue("appearance/fontSize", qBound(8, fontSize, 24));
    apply(themeId, fontFamily, fontSize);
}

void reset() {
    captureDefaults();
    QSettings settings("Afterglow", "Afterglow");
    settings.remove("appearance");
    apply(defaultThemeId(), initialFont.family(), initialFont.pointSize() > 0 ? initialFont.pointSize() : 10);
}

} // namespace Appearance
