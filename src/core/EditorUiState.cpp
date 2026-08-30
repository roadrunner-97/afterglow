#include "EditorUiState.h"

EditorUiState::Mode EditorUiState::mode() const {
    return m_mode;
}

bool EditorUiState::requestMode(Mode requested, bool hasSelection) {
    if (requested != Mode::Gallery && !hasSelection) return false;
    m_mode = requested;
    return true;
}

bool EditorUiState::isProcessing() const {
    return m_processing;
}

void EditorUiState::setProcessing(bool processing) {
    m_processing = processing;
}

bool EditorUiState::isBeforeViewActive() const {
    return m_beforeViewActive;
}

bool EditorUiState::enterBeforeView() {
    if (m_beforeViewActive) return false;
    m_beforeViewActive = true;
    return true;
}

bool EditorUiState::exitBeforeView() {
    if (!m_beforeViewActive) return false;
    m_beforeViewActive = false;
    return true;
}

QString EditorUiState::navigationTarget(const QStringList &paths, const QString &currentPath, int direction) {
    if (paths.isEmpty() || currentPath.isEmpty() || direction == 0) return {};
    const qsizetype index = paths.indexOf(currentPath);
    const qsizetype next  = index + direction;
    if (index < 0 || next < 0 || next >= paths.size()) return {};
    return paths[next];
}
