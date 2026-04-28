#ifndef STYLESHEETS_H
#define STYLESHEETS_H

#include <QString>

// Centralised builders for Qt widget stylesheets used by PhotoEditorApp.
// Each function returns a fully-interpolated stylesheet string for one
// widget — keeping the multi-token arg() chains out of setupUI() and
// removing the brittle %1/%2/%3 ordering scattered through that file.
namespace Stylesheets {

QString toolbar();
QString processingLabel();
QString menuBar();
QString gpuSelectorLabel();
QString gpuSelector();
QString effectPanel();
QString effectTitle();
QString collapseButton();
QString panelSeparator();

} // namespace Stylesheets

#endif // STYLESHEETS_H
