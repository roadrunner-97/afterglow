#ifndef APPEARANCE_H
#define APPEARANCE_H

#include <QList>
#include <QPalette>
#include <QString>

namespace Appearance {

struct ThemeDefinition {
    QString  id;
    QString  name;
    QPalette palette;
    bool     usesSystemPalette;
};

const QList<ThemeDefinition> &themes();
QString                       defaultThemeId();
QString                       savedThemeId();
QString                       savedFontFamily();
int                           savedFontSize();
void                          initialize();
void                          apply(const QString &themeId, const QString &fontFamily, int fontSize);
void                          saveAndApply(const QString &themeId, const QString &fontFamily, int fontSize);
void                          reset();

} // namespace Appearance

#endif
